import re
import time
from collections import Mapping
from glob import glob
from os import makedirs
from os.path import dirname
from os.path import exists as p_exists, join as p_join, isfile as p_isfile, isdir as p_isdir, getctime

import numpy as np
from torch import load as torch_load, save as torch_save
import yaml
import logging
import logging.config


def is_file(path):
    return p_exists(path) and p_isfile(path)


def is_directory(path):
    return p_exists(path) and p_isdir(path)


def make_directories_for_file(path):
    directory = dirname(path)

    if not p_exists(directory):
        makedirs(directory)


def get_htr_logger(name):
    with open('../configs/logging_config.yaml', 'r') as f:
        make_directories_for_file(p_join("../logs/info.log"))
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    return logging.getLogger(name)


logger = get_htr_logger(__name__)


class TimeMeasure(object):
    def __init__(self, enter_msg="", exit_msg="{}.", writer=logger.debug, print_enabled=True):
        self.__enter_msg = enter_msg
        self.__exit_msg = exit_msg
        self.__writer = writer
        self.__time = None
        self.__print_enabled = print_enabled

    def __enter__(self):
        self.__start = time.time()
        if self.__print_enabled and self.__enter_msg:
            self.__writer(self.__enter_msg)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__print_enabled:
            self.__writer(self.__exit_msg.format(pretty_time_interval(self.delta)))

    @property
    def delta(self):
        delta = time.time() - self.__start
        delta = int(delta * 1000)
        return delta


def save_checkpoint(path, total_epochs, model, loss, environment):
    make_directories_for_file(path)
    dictionary = dict()
    dictionary["total_epochs"] = total_epochs
    dictionary["model_states"] = model.state_dict()
    dictionary["loss"] = loss
    dictionary["environment"] = environment.to_dict()
    torch_save(dictionary, path)
    logger.info(f"Saved checkpoint in epoch {total_epochs} to '{path}'.")


def load_checkpoint(path):
    if is_checkpoint(path):
        return torch_load(path)
    else:
        raise RuntimeError("Checkpoint at '{}' does not exist!".format(path))


def is_checkpoint(path: str):
    return is_file(path) and path.endswith(".pt")


def load_latest_checkpoint(directory):
    list_of_files = glob(p_join(directory, '*.pt'))
    latest_file = max(list_of_files, key=getctime)
    return load_checkpoint(latest_file)


class WordDeEnCoder(object):
    def __init__(self, chars):
        self.__chars = chars
        self.__idx_to_char = {i: c for i, c in enumerate(chars)}
        self.__char_to_idx = {v: k for k, v in self.__idx_to_char.items()}

    @property
    def idx_to_char(self):
        return self.__idx_to_char

    @property
    def char_to_idx(self):
        return self.__char_to_idx

    def encode_words(self, words):
        return [np.asarray([self.char_to_idx[letter] for letter in word]) for word in words]

    def decode_word(self, encoded_word):
        return "".join([self.idx_to_char[num] for num in encoded_word])


def inject(value, my_locals):
    if type(value) == str and value.startswith("locals://"):
        path = value.split("//")[1].split("/")
        obj = my_locals[path[0]]
        for i in range(1, len(path)):
            obj = getattr(obj, path[i])
        value = obj

    return value


class FrozenDict(Mapping):
    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)
        self._hash = None

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __hash__(self):
        return hash(tuple(sorted(self._d.items())))


def pretty_time_interval(millis):
    seconds, millis = divmod(millis, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h {minutes}min {seconds}sec {millis}ms"


def build_phrased_regex(left, right):
    return re.compile(r"(?P<left>\|)?" +
                      r"(?P<leftCore>" + left + r")"
                      + r"\|(?P<core>[a-zA-Z0-9.\s|]+)\|"
                      + "(?P<rightCore>" + right + r")"
                      + r"(?P<right>\|)?")


class Replacer(object):
    def __init__(self):
        phrased_regex = build_phrased_regex(left='"', right='"')
        par_regex = build_phrased_regex(left=r'\(', right=r'\)')
        self._regex_pipeline = (phrased_regex, par_regex)

        self._abbreviation = re.compile(r"(\w+\.)\|(\w+)")
        self._left_pipe = re.compile(r"(\|([\s!\\\"#&'()*+,\-./:;?]+))")
        self._right_pipe = re.compile(r"(([\s!\\\"#&'()*+,\-./:;?]+)\|)")

    def __call__(self, line):
        result = line.replace("|,|", ", ")
        for reg in self._regex_pipeline:
            result = reg.sub(r'\g<left> \g<leftCore>\g<core>\g<rightCore>\g<right> ', result)

        result = self._abbreviation.sub(r'\g<1> \g<2>', result)
        result = self._abbreviation.sub(r'\g<1> \g<2>', result)
        result = self._left_pipe.sub(r'\g<2>', result)
        result = self._right_pipe.sub(r'\g<2>', result)
        result = result.replace(") :", "):")
        return result.replace('|', ' ').replace("  ", " ")
