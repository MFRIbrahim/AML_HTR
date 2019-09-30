import Levenshtein
import torch
from torch.nn import functional as F
from torchvision.transforms import transforms

from config import Configuration
from dataset import get_data_loaders
from main import get_available_device, setup_decoder_from_config, build_transformations, build_augmentations, \
    dynamic_learning_rate_small, dynamic_learning_rate_big
from model import get_model_by_name
from pre_processing import pre_processor
from school import TrainingEnvironment, Trainer
from transformations import right_strip, word_tensor_to_list
from util import TimeMeasure, WordDeEnCoder, get_htr_logger
import numpy as np


def main(config_name):
    logger.info(f"Run with config '{config_name}'.")
    with TimeMeasure(enter_msg="Setup everything", exit_msg="Setup finished after {}.", writer=logger.debug):
        device = get_available_device()
        logger.info(f"Active device: {device}")

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
            default=None
        )
        augmentation = transforms.Compose(augmentations) if augmentations is not None else None

        train_loader, train_eval_loader, test_loader = get_data_loaders(meta_path=data_set_config.meta_path,
                                                                        images_path=data_set_config.images_path,
                                                                        transformation=transforms.Compose(
                                                                            transformations),
                                                                        augmentation=augmentation,
                                                                        data_loading_config=data_loading_config,
                                                                        pre_processor=pre_processor(config),
                                                                        max_word_length=data_set_config(
                                                                            "max_word_length"))

        environment = TrainingEnvironment.from_config(environment_config)

        if "small" in model_config.name.lower():
            learning_rate = dynamic_learning_rate_small
        else:
            learning_rate = dynamic_learning_rate_big

        trainer = Trainer(name=training_config.name,
                          model=model,
                          word_prediction=word_predictor_debug,
                          dynamic_learning_rate=learning_rate,
                          environment=environment
                          )

    with TimeMeasure(enter_msg="Load pre-trained model.",
                     exit_msg="Finished loading after {}.",
                     writer=logger.debug):
        model = trainer.load_latest_model()

    with TimeMeasure(writer=logger.debug):
        result = list()

        with torch.no_grad():
            for batch_idx, (feature_batch, label_batch) in enumerate(test_loader):
                feature_batch = feature_batch.to(device)
                label_batch = [right_strip(word, 1.0) for word in word_tensor_to_list(label_batch)]
                label_batch = [de_en_coder.decode_word(word) for word in label_batch]
                model.init_hidden(batch_size=feature_batch.size()[0], device=device)

                output = F.softmax(model(feature_batch), dim=-1)
                output = np.array(output.cpu())
                prediction = word_predictor(output)

                for i in range(len(prediction)):
                    token, target = prediction[i], label_batch[i]
                    character_error_rate = Levenshtein.distance(token, target) / len(target)
                    result.append((target, token, character_error_rate))

        result = sorted(result, key=lambda row: -row[2])
        for idx, (expectation, prediction, error) in enumerate(result):
            logger.info(f"{idx:05d} | {expectation:20s} | {prediction:20s} | {error:6.4f}")


if __name__ == "__main__":
    logger = get_htr_logger(__name__)
    main("config_04")