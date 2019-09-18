import math
import random
from copy import copy
from random import shuffle
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image as PImage
from torch import nn as nn
from torch.nn import functional as F
from torchvision.transforms import transforms

from beam_search import ctcBeamSearch
from data_augmentation import DataAugmenter
from dataset import get_data_loaders
from deslant import deslant_image
from main import load_config, inject, setup_decoder_from_config, build_transformations, \
    get_available_device, get_model_by_name
from model import Net

# Here we use '|' as a symbol the CTC-blank
from school import Trainer, TrainingEnvironment, evaluate_model
from transformations import GrayScale, ToTensor, Rescale, word_tensor_to_list, right_strip
from util import TimeMeasure, WordDeEnCoder

CHAR_LIST = list("| !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
CHAR_DICT = {}
for i in range(len(CHAR_LIST)):
    CHAR_DICT[i] = CHAR_LIST[i]
INV_CHAR_DICT = {v: k for k, v in CHAR_DICT.items()}


def Decoder(matrix):
    # matrix with shape (seq_len, batch_size, num_of_characters) --> (32,50,80)
    C = np.argmax(matrix, axis=2)
    output = []
    # iterate over dim 1 first, since those are the batches
    for i in range(C.shape[1]):
        sub = []
        # iterate over the sequence
        for j in range(C.shape[0]):
            sub.append(CHAR_DICT[C[j][i]])
        output.append(sub)
    for i in range(len(output)):
        output[i] = "".join(output[i])
    return output


def Best_Path_Decoder(matrix):
    # matrix with shape (seq_len, batch_size, num_of_characters) --> (32,50,80)
    C = np.argmax(matrix, axis=2)
    output = []
    # iterate over dim 1 first, since those are the batches
    for i in range(C.shape[1]):
        sub = []
        # iterate over the sequence
        for j in range(C.shape[0]):
            sub.append(CHAR_DICT[C[j][i]])
        output.append(sub)
    # clean the output, i.e. remove multiple letters not seperated by '|' and '|'
    last_letter = "abc"  # invalid label
    current_letter = ""
    output_clean = []
    for i in range(len(output)):
        sub = []
        for j in range(len(output[i])):
            current_letter = output[i][j]
            if output[i][j] != "|" and output[i][j] != last_letter:
                sub.append(output[i][j])
            last_letter = current_letter
        output_clean.append(sub)
    """
    for i in range(len(output)):
        output[i] = "".join(output[i])
    """

    for i in range(len(output_clean)):
        output_clean[i] = "".join(output_clean[i]).strip()
    # print(output)
    return output_clean


def encodeWord(Y):
    new_Y = []
    for w in Y:
        out = []
        for letter in w:
            out.append(int(INV_CHAR_DICT[letter]))
        new_Y.append(np.asarray(out))
    return new_Y


def decodeWord(Y):
    new_Y = []
    for letter in Y:
        new_Y.append(CHAR_DICT[letter])
    return new_Y


if __name__ == "__main__":
    torch.manual_seed(0)
    device = get_available_device()
    print("Active device:", device)

    config_name = "config_01"
    config = load_config("../configs/{}.json".format(config_name))

    prediction_config = SimpleNamespace(**config["prediction"])
    data_set_config = SimpleNamespace(**config["data_set"])
    data_loading_config = SimpleNamespace(**config["data_loading"])
    training_config = SimpleNamespace(**config["training"])
    environment_config = SimpleNamespace(**training_config.environment)
    model_config = SimpleNamespace(**config["model"])

    # in char list we use '|' as a symbol the CTC-blank
    de_en_coder = WordDeEnCoder(list(prediction_config.char_list))
    word_predictor = setup_decoder_from_config(prediction_config, "eval")
    word_predictor_debug = setup_decoder_from_config(prediction_config, "debug")

    retrain_model = True
    # model = Net(dropout=0.2).to(device)
    model = get_model_by_name(model_config.name)(model_config.parameters).to(device)

    transformations = build_transformations(data_loading_config, locals())
    train_loader, test_loader = get_data_loaders(meta_path=data_set_config.meta_path,
                                                 images_path=data_set_config.images_path,
                                                 transformation=transforms.Compose(transformations),
                                                 relative_train_size=data_loading_config.train_size,
                                                 batch_size=data_loading_config.batch_size)


    def dyn_lr(epoch):
        if epoch < 10:
            return 0.01
        elif epoch >= 10:
            return 0.001
        else:
            return 0.00005


    envir = TrainingEnvironment(max_epochs=environment_config.epochs,
                                warm_start=environment_config.warm_start,
                                loss_name=environment_config.loss["name"],
                                optimizer_name=environment_config.optimizer["name"],
                                optimizer_args=environment_config.optimizer["parameters"]
                                )
    trainer = Trainer(training_config.name,
                      word_predictor_debug,
                      dynamic_learning_rate=dyn_lr,
                      environment=envir
                      )



    if retrain_model:
        trainer.train(model, train_loader, device=device)

    my_locals = locals()
    evals = [(eval_obj["name"], inject(eval_obj["data_loader"], my_locals)) for eval_obj in config["evaluation"]]

    with TimeMeasure(enter_msg="Evaluate model:", exit_msg="Evaluation finished after {} ms."):
        for name, loader in evals:
            evaluate_model(msg=name + " accuracy: {:7.4f}",
                           word_prediction=word_predictor,
                           de_en_coder=de_en_coder,
                           model=model,
                           data_loader=loader,
                           device=device
                           )
