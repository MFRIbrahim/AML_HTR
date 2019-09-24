import logging
import threading

import PIL
import numpy as np
import torch
from cv2 import cvtColor, COLOR_BGR2GRAY, resize
from torch import Tensor as TorchTensor
from torchvision import transforms

from util import inject, get_htr_logger

logger = get_htr_logger(__name__)


def transformation_from_entry(entry, my_locals):
    if type(entry) == str:
        transform = get_transformation_by_name(entry)
        result = transform(dict())
    else:
        transform = get_transformation_by_name(entry["name"])
        parameters = {k: inject(v, my_locals) for k, v in entry.get("parameters", dict()).items()}
        result = transform(parameters)

    return result


def get_transformation_by_name(name):
    if name == "GrayScale":
        return lambda params: GrayScale()
    elif name == "Rescale":
        return lambda params: Rescale(**params)
    elif name == "ToTensor":
        return lambda params: ToTensor(**params)
    elif name == "TensorToPIL":
        return lambda params: TensorToPIL(**params)
    elif name == "RandomErasing":
        return lambda params: RandomErasing(**params)
    elif name == "RandomRotateAndTranslate":
        return lambda params: RandomRotateAndTranslate(**params)
    elif name == "RandomJitter":
        return lambda params: RandomJitter(**params)
    elif name == "RandomPerspective":
        return lambda params: RandomPerspective(**params)
    elif name == "TensorToNumpy":
        return lambda params: TensorToNumpy(**params)
    elif name == "PadTranscript":
        return lambda params: PadTranscript(**params)
    else:
        raise RuntimeError(f"Didn't find transformation by name '{name}'")


class GrayScale(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, transcript = sample["image"], sample["transcript"]
        gray = cvtColor(image, COLOR_BGR2GRAY)
        return {"image": gray, "transcript": transcript}


class PadTranscript(object):
    def __init__(self, max_word_length):
        self.__max_word_length = max_word_length

    def __call__(self, sample):
        image, transcript = sample["image"], sample["transcript"]
        padded_transcript = (transcript + (self.__max_word_length - len(transcript))* " ")
        return {"image": image, "transcript": padded_transcript}


class Rescale(object):
    def __init__(self, new_height, new_width):
        self.__new_height = new_height
        self.__new_width = new_width

    def __call__(self, sample):
        image, transcript = sample["image"], sample["transcript"]
        scaled_image = resize_embedded(image, (self.__new_width, self.__new_height))
        return {"image": scaled_image, "transcript": transcript}


class ToTensor(object):
    def __init__(self, char_to_int):
        self.__converter = transforms.ToTensor()
        self.__char_to_int = char_to_int

    def __call__(self, sample):
        image, transcript = sample["image"], sample["transcript"]
        tensor = self.__converter(image)
        word = transcript
        if type(transcript) == str:
            word = [self.__char_to_int[letter] for letter in transcript]
        return tensor.float(), TorchTensor(word)


class TensorToPIL(object):
    def __init__(self):
        self.__transform = transforms.ToPILImage("L")
        self.unsqueezed = False

    def __call__(self, sample):
        image, transcript = sample
        if not (type(image) == torch.Tensor):
            raise ValueError(f"Can only transform torch.Tensor to PIL Image, not  '{type(image)}'")
        if image.ndim == 2:
            image = image.unsqueeze(0)
            self.unsqueezed = True
        return self.__transform(image), transcript


class TensorToNumpy(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, transcript = sample
        if not (type(image) == torch.Tensor):
            raise ValueError(f"Can only transform torch.Tensor to Numpy Array, not  '{type(image)}'")
        return self.to_numpy(image), transcript

    def to_numpy(self, sample):
        return np.asarray(sample)


class RandomErasing(object):
    def __init__(self, p=0.1, scale=(0.02, 0.04), ratio=(0.3, 3.3), value=1):
        self.__transform = transforms.RandomErasing(p=p, scale=scale, ratio=ratio, value=value)

    def __call__(self, sample):
        image, transcript = sample
        if not (type(image) == torch.Tensor):
            raise ValueError(f"Can only perform random erasing torch.Tensor, not  '{type(image)}'")
        return self.__transform(image), transcript


class RandomRotateAndTranslate(object):
    def __init__(self, p=0.1, degrees=0, translate=(0.03, 0.03), fillcolor=255):
        self.__transform = transforms.RandomApply(
            [transforms.RandomAffine(degrees=degrees, translate=translate, fillcolor=fillcolor)], p=p)

    def __call__(self, sample):
        image, transcript = sample
        if not (type(image) == PIL.Image.Image):
            raise ValueError(f"Can only perform Rotation and Translation on PIL.Image.Image, not  '{type(image)}'")
        return self.__transform(image), transcript


class RandomJitter(object):
    def __init__(self, p=0.1):
        self.__transform = transforms.RandomApply([transforms.ColorJitter()], p=p)

    def __call__(self, sample):
        image, transcript = sample
        if not (type(image) == PIL.Image.Image):
            raise ValueError(f"Can only perform Jitter on PIL.Image.Image, not  '{type(image)}'")
        return self.__transform(image), transcript


class RandomPerspective(object):

    def __init__(self, p=0.1, warp_ratio=0.0003, fillcolor=255):
        self.p = p
        self.warp_ratio = warp_ratio
        self.fillcolor = fillcolor

    def __call__(self, sample):
        image, transcript = sample
        if not (type(image) == PIL.Image.Image):
            raise ValueError("Can only perform random perspective on PIL.Image.Image, not  '{}'".format(type(image)))
        return {"image": self.__warp(image), "transcript": transcript}

    def __warp(self, img):
        try:
            if np.random.rand() > self.p:
                return img
            pa = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
            pb = pa + torch.randn(4, 2) * self.warp_ratio

            result = img.transform(img.size,
                                   PIL.Image.PERSPECTIVE,
                                   RandomPerspective.__find_coefficients(pa, pb),
                                   PIL.Image.BICUBIC,
                                   fillcolor=self.fillcolor)

        except np.linalg.LinAlgError:
            result = img

        return result

    @staticmethod
    def __find_coefficients(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        # np.matrix needed, because otherwise 10% of the time it crashes, Reason: Singular Matrix
        A = np.matrix(matrix, dtype=np.float)
        B = np.array(pb).reshape(8)

        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)


def right_strip(lst, value):
    for idx, x in enumerate(reversed(lst)):
        if x != value:
            if idx:
                del lst[-idx:]
            return lst
    return lst


def word_tensor_to_list(tensor):
    return [right_strip(word, 0) for word in tensor.cpu().tolist()]

def resize_embedded(img, size):
    width, height = size
    (h, w) = img.shape
    fx = w / width
    fy = h / height
    f = max(fx, fy)
    new_size = (max(min(width, int(w / f)), 1), max(min(height, int(h / f)), 1))
    img = resize(img, (new_size[0], new_size[1]))
    target = np.ones([height, width]) * 255
    target[0:new_size[1], 0:new_size[0]] = img
    return target
