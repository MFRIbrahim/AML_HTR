from torch.utils.data import Dataset, DataLoader, random_split
from json import dump as json_write, load as json_read
from torch import Tensor as TorchTensor
import cv2
import numpy as np
import os

from util import TimeMeasure, is_file


class WordsDataSet(Dataset):
    __health_state = "health_state.json"

    def __init__(self, meta_file, root_dir, transform=None):
        self.__meta_file = meta_file
        self.__words = list()
        self.__root_dir = root_dir
        self.__transform = transform
        self.__statistics = None

        with TimeMeasure(enter_msg="Begin meta data loading.",
                         exit_msg="Finished meta data loading after {} ms.",
                         writer=print):
            self.__process_meta_file()
            self.__availability_check()

        with TimeMeasure(enter_msg="Begin health check.",
                         exit_msg="Finished health check after {} ms.",
                         writer=print):
            self.__health_check()

        with TimeMeasure(enter_msg="Begin creating statistics.",
                         exit_msg="Finished creating statistics after {} ms.",
                         writer=print):
            self.__create_statistics()

    def __process_meta_file(self):
        with open(self.__meta_file, 'r') as fp:
            for line in fp:
                self.__process_meta_line(line)

    def __process_meta_line(self, line):
        if not line.startswith("#"):
            self.__words.append(WordsMetaData.parse(line))

    def __availability_check(self):
        to_delete = []
        for idx, word_meta in enumerate(self.__words):
            path = word_meta.path(self.__root_dir)
            if not is_file(path):
                print("File not found:", path)
                to_delete.append(idx)

        self.__save_delete_indices(to_delete)

    def __save_delete_indices(self, to_delete):
        for idx in sorted(to_delete, key=lambda x: -x):
            del self.__words[idx]

    def __health_check(self):
        to_delete = list()
        health_path = os.path.join(self.__root_dir, WordsDataSet.__health_state)
        if is_file(health_path):
            with open(health_path, 'r') as fp:
                to_delete = json_read(fp)
        else:
            for idx, word_meta in enumerate(self.__words):
                try:
                    self[idx]
                except cv2.error:
                    to_delete.append(idx)
            print("Write corrupted indices to '{}'".format(health_path))
            with open(health_path, 'w') as fp:
                json_write(to_delete, fp)

        print("WordsDataSet - Health Check: {} indices={} not readable.".format(len(to_delete), to_delete))
        self.__save_delete_indices(to_delete)

    def __create_statistics(self):
        min_length, max_length, summed_length = np.inf, 0, 0
        min_id, max_id = "", ""
        min_word, max_word = "", ""
        for word_meta in self.__words:
            length = len(word_meta.transcription)
            if length < min_length:
                min_length = length
                min_word = '"' + word_meta.transcription + '"'
                min_id = word_meta.word_id

            if max_length < length:
                max_length = length
                max_word = '"' + word_meta.transcription + '"'
                max_id = word_meta.word_id

            summed_length += length

        self.__statistics = {"min_length": min_length,
                             "max_length": max_length,
                             "avg_length": summed_length/len(self.__words),
                             "min_id": min_id,
                             "max_id": max_id,
                             "min_word": min_word,
                             "max_word": max_word
                             }

    def __len__(self):
        return len(self.__words)

    def __getitem__(self, idx):
        if type(idx) == TorchTensor:
            idx = idx.item()
        meta = self.__words[idx]
        path = meta.path(self.__root_dir)
        image = cv2.imread(path)
        sample = {"image": image, "transcript": meta.transcription}

        if self.__transform is not None:
            sample = self.__transform(sample)

        return sample

    @property
    def statistics(self):
        return self.__statistics


class BoundingBox(object):
    def __init__(self, x, y, w, h):
        self.__x = x
        self.__y = y
        self.__w = w
        self.__h = h

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def pos(self):
        return self.x, self.y

    @property
    def w(self):
        return self.__w

    @property
    def width(self):
        return self.w

    @property
    def h(self):
        return self.__h

    @property
    def height(self):
        return self.h


class WordsMetaData(object):
    def __init__(self, wid, segmentation_state, gray_level, bounding_box, pos_tag, transcription):
        """

        :param wid: word id
        :param segmentation_state: 'ok' - word was correctly, 'err' - segmentation of word can be bad
        :param gray_level: graylevel to binarize the line containing this word
        :param bounding_box: bounding box around this word
        :param pos_tag: the grammatical tag for this word
        :param transcription: the transcription for this word
        """
        self.__wid = wid
        self.__segmentation_state = segmentation_state
        self.__gray_level = gray_level
        self.__bounding_box = bounding_box
        self.__pos_tag = pos_tag
        self.__transcription = transcription

    @property
    def word_id(self):
        return self.__wid

    @property
    def segmentation_state(self):
        return self.__segmentation_state

    @property
    def gray_level(self):
        return self.gray_level

    @property
    def bounding_box(self):
        return self.__bounding_box

    @property
    def pos_tag(self):
        return self.__pos_tag

    @property
    def transcription(self):
        return self.__transcription

    def path(self, root):
        wid = self.word_id
        folder, subfolder, counter, sub_counter = wid.split("-")
        return os.path.join(root, folder, folder + "-" + subfolder, wid + ".png")

    @staticmethod
    def parse(line):
        line = line.strip()
        parts = line.split(" ")
        wid = parts[0]
        state = parts[1]
        gray_level = parts[2]
        pos_tag = parts[7]
        transcription = parts[8]
        box = BoundingBox(x=parts[3], y=parts[4], w=parts[5], h=parts[6])
        return WordsMetaData(wid, state, gray_level, box, pos_tag, transcription)


def get_data_loaders(meta_path, images_path, transformation, relative_train_size, batch_size):
    with TimeMeasure(enter_msg="Begin initialization of data set.",
                     exit_msg="Finished initialization of data set after {} ms.",
                     writer=print):
        data_set = WordsDataSet(meta_path, images_path, transform=transformation)

    with TimeMeasure(enter_msg="Splitting data set", writer=print):
        train_size = int(relative_train_size * len(data_set))
        test_size = len(data_set) - train_size
        train_data_set, test_data_set = random_split(data_set, (train_size, test_size))

    with TimeMeasure(enter_msg="Init data loader", writer=print):
        train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
        test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    return train_loader, test_loader
