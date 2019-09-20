import logging
import os
from json import dump as json_write, load as json_read

import cv2
import dill
import numpy as np
from torch import Tensor as TorchTensor
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold

from util import TimeMeasure, is_file, make_directories_for_file

logger = logging.getLogger(__name__)


class WordsDataSet(Dataset):
    __health_state = "health_state{}.json"

    def __init__(self, meta_file, root_dir, transform=None, pre_processor=None):
        self.__meta_file = meta_file
        self.__words = list()
        self.__root_dir = root_dir
        self.__transform = transform
        self.__statistics = None
        self.__pre_processor = pre_processor

        if self.__pre_processor is None:
            logger.info("No pre-processor selected.")
        else:
            logger.info(f"Selected pre-processor: {pre_processor.name}")

        with TimeMeasure(enter_msg="Begin meta data loading.",
                         exit_msg="Finished meta data loading after {}.",
                         writer=logger.debug):
            self.__process_meta_file()
            self.__availability_check()

        with TimeMeasure(enter_msg="Begin health check.",
                         exit_msg="Finished health check after {}.",
                         writer=logger.debug):
            self.__health_check()

        with TimeMeasure(enter_msg="Begin creating statistics.",
                         exit_msg="Finished creating statistics after {}.",
                         writer=logger.debug):
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
                logger.warning("File not found:", path)
                to_delete.append(idx)

        self.__save_delete_indices(to_delete)

    def __save_delete_indices(self, to_delete):
        for idx in sorted(to_delete, key=lambda x: -x):
            del self.__words[idx]

    def __health_check(self):
        to_delete = list()
        if self.__pre_processor is None:
            health_state = WordsDataSet.__health_state.format("")
        else:
            health_state = WordsDataSet.__health_state.format(f"_{self.__pre_processor.name}")

        health_path = os.path.join(self.__root_dir, health_state)
        if is_file(health_path):
            with open(health_path, 'r') as fp:
                to_delete = json_read(fp)
        else:
            for idx, word_meta in enumerate(self.__words):
                try:
                    self[idx]
                except (cv2.error, ValueError) as e:
                    logger.error(f"Corrupted file at index: {idx}")
                    to_delete.append(idx)
            logger.debug(f"Write corrupted indices to '{health_path}'")
            with open(health_path, 'w') as fp:
                json_write(to_delete, fp)

        logger.info(f"WordsDataSet - Health Check: {len(to_delete)} indices={to_delete} not readable.")
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
                             "avg_length": summed_length / len(self.__words),
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

        if self.__pre_processor is None:
            path = meta.path(self.__root_dir)
            image = cv2.imread(path)
        else:
            image = self.__pre_processor(meta, self.__root_dir)

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


def get_data_loaders(meta_path, images_path, transformation, augmentation, data_loading_config, pre_processor=None):
    relative_train_size = data_loading_config.train_size
    batch_size = data_loading_config.batch_size
    restore_path = data_loading_config.get("restore_path", default=None)
    save_path = data_loading_config.get("save_path", default=None)

    with TimeMeasure(enter_msg="Begin initialization of data set.",
                     exit_msg="Finished initialization of data set after {}.",
                     writer=logger.debug):
        data_set = WordsDataSet(meta_path, images_path, transform=transformation, pre_processor=pre_processor)

    with TimeMeasure(enter_msg="Splitting data set", writer=logger.debug):
        if restore_path is not None and os.path.exists(restore_path):
            loaded = True
            train_data_set, test_data_set = __restore_train_test_split(restore_path, data_set)
        else:
            loaded = False
            train_size = int(relative_train_size * len(data_set))
            test_size = len(data_set) - train_size
            train_data_set, test_data_set = random_split(data_set, (train_size, test_size))

        if augmentation is not None:
            train_data_set = AugmentedDataSet(train_data_set, augmentation)

    if not loaded:
        __save_train_test_split(save_path, train_data_set, test_data_set)

    with TimeMeasure(enter_msg="Init data loader", writer=logger.debug):
        train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)
        test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)

    return train_loader, test_loader


def get_data_loaders_cv(meta_path,
                        images_path,
                        transformation,
                        augmentation,
                        data_loading_config,
                        pre_processor=None,
                        number_of_splits=3):
    batch_size = data_loading_config.batch_size

    with TimeMeasure(enter_msg="Begin initialization of data set.",
                     exit_msg="Finished initialization of data set after {}.",
                     writer=logger.debug):
        data_set = WordsDataSet(meta_path, images_path, transform=transformation, pre_processor=pre_processor)

    with TimeMeasure(enter_msg="Splitting data set", writer=logger.debug):
        train_test_array = cv_split(data_set, number_of_splits, augmentation)

    with TimeMeasure(enter_msg="Init data loader", writer=logger.debug):
        loader_array = []
        for train_set, test_set, augmented_set in train_test_array:
            train_eval_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)
            train_loader = DataLoader(augmented_set, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)
            loader_array.append((train_loader, train_eval_loader, test_loader))

    return loader_array


def __save_train_test_split(path, train_data_set, test_data_set):
    make_directories_for_file(path)
    with open(path, "wb") as fp:
        dill.dump([train_data_set.indices, test_data_set.indices], fp)


def __restore_train_test_split(path, data_set):
    with open(path, "rb") as fp:
        splits = dill.load(fp)

    if type(splits) != list:
        msg = f"Unknown datatype for splits: '{type(splits)}', has to be list"
        logger.critical(msg)
        raise ValueError(msg)

    if len(splits) != 2:
        msg = f"Expected splits to have length 2, not '{len(splits)}'"
        logger.critical(msg)
        raise ValueError(msg)

    train_data_set = Subset(data_set, splits[0])
    test_data_set = Subset(data_set, splits[1])
    return train_data_set, test_data_set


class AugmentedDataSet(Dataset):
    def __init__(self, source, augmentation):
        self.__source = source
        self.__augmentation = augmentation

    def __len__(self):
        return len(self.__source)

    def __getitem__(self, idx):
        return self.__augmentation(self.__source[idx])

    @property
    def indices(self):
        return self.__source.indices


def cv_split(dataset, n, augmentation=None):
    """
    Split the dataset into n non-overlapping new datasets where one is used for testing and
    return an array that contains the sequence of splits.

    Arguments:
        dataset (Dataset): Dataset to be split
        n (int): number of non-overlapping new datasets
        augmentation : augmentations
    """
    cv = KFold(n_splits=n, random_state=0)
    res = []

    for train_index, test_index in cv.split(dataset):
        train_set = Subset(dataset, train_index)
        test_set = Subset(dataset, test_index)

        if augmentation is not None:
            augmented_set = AugmentedDataSet(train_set, augmentation)

        res.append((train_set, test_set, augmented_set))

    return res