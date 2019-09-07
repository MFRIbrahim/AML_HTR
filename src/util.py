import time
from glob import glob
from os import makedirs
from os.path import dirname
from os.path import exists as p_exists, join as p_join, isfile as p_isfile, isdir as p_isdir, getctime

import numpy as np
from torch import load as torch_load, save as torch_save


def is_file(path):
    return p_exists(path) and p_isfile(path)


def is_directory(path):
    return p_exists(path) and p_isdir(path)


def make_directories_for_file(path):
    directory = dirname(path)

    if not p_exists(directory):
        makedirs(directory)


class TimeMeasure(object):
    def __init__(self, enter_msg="", exit_msg="{} ms.", writer=print, print_enabled=True):
        self.__enter_msg = enter_msg
        self.__exit_msg = exit_msg
        self.__writer = writer
        self.__time = None
        self.__print_enabled = print_enabled

    def __enter__(self):
        self.__start = time.time()
        if self.__print_enabled and self.__enter_msg:
            self.__writer(self.__enter_msg)

    def __exit__(self, exc_type, exc_val, exc_tb):
        delta = time.time() - self.__start
        delta = int(delta * 1000)
        if self.__print_enabled:
            self.__writer(self.__exit_msg.format(delta))


def save_checkpoint(path, total_epochs, model, loss, environment):
    make_directories_for_file(path)
    dictionary = dict()
    dictionary["total_epochs"] = total_epochs
    dictionary["model_states"] = model.state_dict()
    dictionary["loss"] = loss
    dictionary["environment"] = environment.to_dict()
    torch_save(dictionary, path)


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
