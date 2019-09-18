import os

import cv2
import numpy as np

from deslant import deslant_image
from util import FrozenDict, is_file, make_directories_for_file


def pre_processor(config):
    name = config.get("data_set/pre_processor/name", default=None)
    parameters = config.get("data_set/pre_processor/parameters", default=FrozenDict())

    if name == "Deslant":
        return DeslantPreprocessor(name, params=parameters)
    else:
        raise None


class DeslantPreprocessor(object):
    def __init__(self, name, params=FrozenDict()):
        self.__name = name
        self.__transform = Deslant(**params)

    @property
    def name(self):
        return self.__name

    def __call__(self, meta, img_directory):
        root = os.path.dirname(img_directory)
        deslant_path = meta.path(os.path.join(root, self.name))

        if is_file(deslant_path):
            result = cv2.imread(deslant_path)
        else:
            make_directories_for_file(deslant_path)
            original_path = meta.path(img_directory)
            image = cv2.imread(original_path)
            result = self.__transform(image)
            cv2.imwrite(deslant_path, image)

        return result


class Deslant(object):
    def __init__(self, fillcolor=255, alpha_values=(-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3)):
        self.fillcolor = fillcolor
        self.alpha_values = alpha_values

    def __call__(self, image):
        if not (type(image) == np.ndarray):
            raise ValueError(f"Can only perform deslanting on np.ndarray, not '{type(image)}'")

        return deslant_image(image, bgcolor=self.fillcolor, alpha_vals=self.alpha_values)
