import json

import torch
from torchvision import transforms

from dataset import get_data_loaders
from model import Net
from school import TrainingEnvironment, Trainer, evaluate_model
from statistics import Statistics
from transformations import GrayScale, Rescale, ToTensor, RandomErasing, RandomJitter, RandomRotateAndTranslate, RandomPerspective, Deslant, TensorToNumpy, TensorToPIL, PadTranscript
from util import WordDeEnCoder, TimeMeasure
from word_prediction import BeamDecoder, BestPathDecoder, SimpleWordDecoder
from types import SimpleNamespace
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


def load_config(path):
    with open(path, 'r') as fp:
        return json.load(fp)


def get_decoder_by_name(name):
    if name == "Simple":
        return lambda params: SimpleWordDecoder(params["char_list"])
    elif name == "BestPath":
        return lambda params: BestPathDecoder(params["char_list"])
    elif name == "Beam":
        return lambda params: BeamDecoder(params["char_list"])
    else:
        raise RuntimeError("Didn't find decoder by name '{}'".format(name))


def setup_decoder_from_config(config, category):
    base_parameters = {"char_list": list(config.char_list)}
    decoder_config = config.word_prediction[category]

    name = decoder_config["name"]
    decoder = get_decoder_by_name(name)
    parameters = decoder_config.get("parameters", {})
    parameters = {**base_parameters, **parameters}

    return decoder(parameters)

def create_augmented_data_set(transformations, config):
    #if there are no augmentations to be performed, bail
    if not hasattr(config, "augmented_meta_path") or not hasattr(config, "augmented_image_path"):
        return config
    augmented_meta_path = config.augmented_meta_path
    if not os.path.exists(augmented_meta_path):
        os.makedirs(augmented_meta_path)
    augmented_image_path = config.augmented_image_path
    if not os.path.exists(augmented_image_path):
        os.makedirs(augmented_image_path)
    meta_path = config.meta_path
    images_path = config.images_path
    line_count = 0
    with open(meta_path) as f:
        with open(os.path.join(augmented_meta_path, meta_path.split('/')[-1]), "w+") as f_new:
            for line in f:
                line_count += 1
                # skip empty lines and information at the beginning
                if not line.strip() or line[0] == "#":
                    f_new.write(line)
                    continue
                for i, perm in enumerate(transformations):
                    # construct the image path from the information in the corresponding words.txt lines
                    line_split = line.strip().split(' ')
                    file_name_split = line_split[0].split('-')
                    file_name = '/' + file_name_split[0] + '/' + file_name_split[0] + '-' + file_name_split[1] + '/' + line_split[0] + '.png'
                    # load image, resize to desired image size, convert to greyscale and then to torch tensor
                    try:
                        img = cv2.imread(images_path + file_name)
                    except Exception as e:
                        continue
                    img = transforms.Compose(perm)({"image": img, "transcript": line_split[-1]})["image"]
                    #create new file name to indicate permutation
                    new_file_name_split = line_split[0].split('-')
                    new_line_split = deepcopy(line_split)
                    new_line_split[0] = new_line_split[0] + "aug" + str(i)
                    new_file_folder ='/' + new_file_name_split[0] + '/' + new_file_name_split[0] + '-' + new_file_name_split[1] + '/'
                    new_file_name =  new_file_folder + new_line_split[0] + '.png'
                    if not os.path.exists(augmented_image_path + new_file_folder):
                        os.makedirs(augmented_image_path + new_file_folder)

                    new_line_split.append('\n')
                    new_line = " ".join(new_line_split)

                    cv2.imwrite(augmented_image_path + new_file_name, (img*255).astype(np.uint8))
                    f_new.write(new_line)
    config.meta_path = os.path.join(augmented_meta_path, meta_path.split('/')[-1])
    print(config.meta_path)
    config.images_path = augmented_image_path
    return config


def create_transformations_from_config(config, my_locals):
    result = list()
    if hasattr(config, "transformations"):
        for entry in config.transformations:
                transform = get_transformation_by_name(entry["name"])
                parameters = {k: inject(v, my_locals) for k, v in entry.get("parameters", dict()).items()}
                result.append(transform(parameters))
    elif hasattr(config, "permutations"):
        for perm in config.permutations:
            transformations = list()
            for entry in list(perm.values())[0]:
                transform = get_transformation_by_name(entry["name"])
                parameters = {k: inject(v, my_locals) for k, v in entry.get("parameters", dict()).items()}
                transformations.append(transform(parameters))
            result.append(transformations)
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
    elif name == "Deslant":
        return lambda params: Deslant(**params)
    elif name == "TensorToNumpy":
        return lambda params: TensorToNumpy(**params)
    elif name == "PadTranscript":
        return lambda params: PadTranscript(**params)
    else:
        raise RuntimeError("Didn't find transformation by name '{}'".format(name))


def inject(value, my_locals):
    if type(value) == str and value.startswith("locals://"):
        path = value.split("//")[1].split("/")
        obj = my_locals[path[0]]
        for i in range(1, len(path)):
            obj = getattr(obj, path[i])
        value = obj

    return value


def get_model_by_name(name):
    if name == "Net":
        return lambda params: Net(**params)
    else:
        raise RuntimeError("Unknown specified network '{}'".format(name))


def main(config_name):
    with TimeMeasure(enter_msg="Setup everything", exit_msg="Setup finished after {} ms."):
        torch.manual_seed(0)
        device = get_available_device()
        print("Active device:", device)

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

        model = get_model_by_name(model_config.name)(model_config.parameters).to(device)

        transformations = create_transformations_from_config(data_loading_config, locals())

        data_augmentations = create_transformations_from_config(data_set_config, locals())

        data_set_config = create_augmented_data_set(data_augmentations, data_set_config)

        split_restore_path, split_save_path = None, None
        if hasattr(data_loading_config, "restore_path"):
            split_restore_path = data_loading_config.restore_path
        if hasattr(data_loading_config, "save_path"):
            split_save_path = data_loading_config.save_path

        train_loader, test_loader = get_data_loaders(meta_path=data_set_config.meta_path,
                                                     images_path=data_set_config.images_path,
                                                     transformation=transforms.Compose(transformations),
                                                     relative_train_size=data_loading_config.train_size,
                                                     batch_size=data_loading_config.batch_size,
                                                     restore_path=split_restore_path,
                                                     save_path=split_save_path)

        environment = TrainingEnvironment(max_epochs=environment_config.epochs,
                                          warm_start=environment_config.warm_start,
                                          loss_name=environment_config.loss["name"],
                                          optimizer_name=environment_config.optimizer["name"],
                                          optimizer_args=environment_config.optimizer["parameters"]
                                          )

        trainer = Trainer(training_config.name,
                          word_predictor_debug,
                          dynamic_learning_rate=dynamic_learning_rate,
                          environment=environment
                          )
        stats = Statistics.get_instance(training_config.name)
        stats.reset()

        my_locals = locals()
        evals = [(eval_obj["name"], inject(eval_obj["data_loader"], my_locals)) for eval_obj in config["evaluation"]]

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
