import json

import torch
from torchvision import transforms

from config import Configuration
from dataset import get_data_loaders
from model import get_model_by_name
from pre_processing import pre_processor
from school import TrainingEnvironment, Trainer, evaluate_model
from statistics import Statistics
from transformations import get_transformation_by_name, transformation_from_entry
from util import WordDeEnCoder, TimeMeasure, inject
from word_prediction import get_decoder_by_name
from copy import deepcopy
import os
import cv2
import numpy as np


def get_available_device():
    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = 'cpu'

    return device


def dynamic_learning_rate(epoch):
    if epoch < 10:
        return 0.01
    elif epoch >= 10:
        return 0.001
    else:
        return 0.00005


def setup_decoder_from_config(config, category):
    base_parameters = {"char_list": list(config.char_list)}
    decoder_config = config[f"word_prediction/{category}"]

    name = decoder_config.name
    decoder = get_decoder_by_name(name)
    parameters = decoder_config.get("parameters", default={})
    parameters = {**base_parameters, **parameters}

    return decoder(parameters)


def build_transformations(transformations, my_locals):
    return [transformation_from_entry(entry, my_locals) for entry in transformations]


def build_augmentations(augmentations, my_locals):
    pass


def main(config_name):
    with TimeMeasure(enter_msg="Setup everything", exit_msg="Setup finished after {} ms."):
        torch.manual_seed(0)
        device = get_available_device()
        print("Active device:", device)

        config = Configuration(f"../configs/{config_name}.json")

        prediction_config = config["prediction"]
        data_set_config = config["data_set"]
        data_loading_config = config["data_loading"]
        training_config = config["training"]
        environment_config = config["training/environment"]
        model_config = config["model"]

        # in char list we use '|' as a symbol the CTC-blank
        de_en_coder = WordDeEnCoder(list(prediction_config.char_list))
        word_predictor = setup_decoder_from_config(prediction_config, "eval")
        word_predictor_debug = setup_decoder_from_config(prediction_config, "debug")

        model = get_model_by_name(model_config.name)(model_config.parameters).to(device)

        main_locals = locals()
        transformations = build_transformations(data_loading_config.transformations, main_locals)
        augmentations = data_loading_config.if_exists(
            path="augmentations",
            runner=lambda augms: build_augmentations(augms, main_locals),
            default=list()
        )

        train_loader, test_loader = get_data_loaders(meta_path=data_set_config.meta_path,
                                                     images_path=data_set_config.images_path,
                                                     transformation=transforms.Compose(transformations),
                                                     augmentation=transforms.Compose(augmentations),
                                                     data_loading_config=data_loading_config,
                                                     pre_processor=pre_processor(config))

        environment = TrainingEnvironment.from_config(environment_config)

        trainer = Trainer(training_config.name,
                          word_predictor_debug,
                          dynamic_learning_rate=dynamic_learning_rate,
                          environment=environment
                          )
        stats = Statistics.get_instance(training_config.name)
        stats.reset()

        my_locals = locals()
        evals = [(eval_obj["name"], inject(eval_obj["data_loader"], my_locals)) for eval_obj in config("evaluation")]

    def run_model_evaluation():
        with TimeMeasure(enter_msg="Evaluate model:", exit_msg="Evaluation finished after {} ms."):
            result = dict()
            for name, loader in evals:
                metrics = evaluate_model(word_prediction=word_predictor,
                                         de_en_coder=de_en_coder,
                                         model=model,
                                         data_loader=loader,
                                         device=device
                                         )
                for k in metrics.keys():
                    print(f"{name} {k}: {metrics[k]:7.4f}")
                result[name] = metrics
        return result

    trainer.model_eval = run_model_evaluation

    with TimeMeasure(enter_msg="Get trained model.", exit_msg="Obtained trained model after {} ms."):
        if training_config.retrain:
            with TimeMeasure(enter_msg="Begin Training.", exit_msg="Finished complete training after {} ms."):
                trainer.train(model, train_loader, device=device)
        else:
            with TimeMeasure(enter_msg="Load pre-trained model.", exit_msg="Finished loading after {} ms."):
                trainer.load_latest_model_state_into(model)


if __name__ == "__main__":
    main("config_04")
