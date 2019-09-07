from torchvision import transforms
from cv2 import cvtColor, COLOR_BGR2GRAY, resize, imshow, waitKey
from torch import Tensor as TorchTensor


class GrayScale(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, transcript = sample["image"], sample["transcript"]
        gray = cvtColor(image, COLOR_BGR2GRAY)
        return {"image": gray, "transcript": transcript}


class Rescale(object):
    def __init__(self, new_width, new_height, max_word_length):
        self.__new_width = new_width
        self.__new_height = new_height
        self.__max_word_length = max_word_length

    def __call__(self, sample):
        image, transcript = sample["image"], sample["transcript"]
        scaled_image = resize(image, (self.__new_height, self.__new_width))
        scaled_transcript = (transcript + self.__max_word_length*" ")[:self.__max_word_length]
        return {"image": scaled_image, "transcript": scaled_transcript}


class ToTensor(object):
    def __init__(self, char_to_int):
        self.__converter = transforms.ToTensor()
        self.__char_to_int = char_to_int

    def __call__(self, sample):
        image, transcript = sample["image"], sample["transcript"]
        tensor = self.__converter(image)
        word = [self.__char_to_int[letter] for letter in transcript]

        return tensor, TorchTensor(word)


def rstrip(lst, value):
    for idx, x in enumerate(reversed(lst)):
        if x != value:
            if idx:
                del lst[-idx:]
            return lst
    return lst


def word_tensor_to_list(tensor):
    return [rstrip(word, 0) for word in tensor.cpu().tolist()]

