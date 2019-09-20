import torch
from torchvision import transforms

from config import Configuration
from dataset import get_data_loaders
from dataset import get_data_loaders_cv
from model import get_model_by_name
from pre_processing import pre_processor
from school import TrainingEnvironment, Trainer, KfoldTrainer, evaluate_model
from statistics import Statistics
from transformations import transformation_from_entry
from util import WordDeEnCoder, TimeMeasure, inject, get_htr_logger
from word_prediction import get_decoder_by_name


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
    def transformation(entry):
        return transformation_from_entry(entry, my_locals)

    result = list()

    for augmentation_block in augmentations:
        transformations = list()
        if "pre" in augmentation_block:
            transformations.append(transformation(augmentation_block["pre"]))

        transformations.extend([transformation(entry) for entry in augmentation_block["transformations"]])

        if "post" in augmentation_block:
            transformations.append(transformation(augmentation_block["post"]))

        result.extend(transformations)

    return result


def main2(config_name):
    logger.info(f"Run with config '{config_name}'.")
    with TimeMeasure(enter_msg="Setup everything", exit_msg="Setup finished after {}.", writer=logger.debug):
        torch.manual_seed(0)
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

        main_locals = locals()
        transformations = build_transformations(data_loading_config.transformations, main_locals)
        augmentations = data_loading_config.if_exists(
            path="augmentations",
            runner=lambda augms: build_augmentations(augms, main_locals),
            default=None
        )
        augmentation = transforms.Compose(augmentations) if augmentations is not None else None

        loader_array = get_data_loaders_cv(meta_path=data_set_config.meta_path,
                                                     images_path=data_set_config.images_path,
                                                     transformation=transforms.Compose(transformations),
                                                     augmentation=augmentation,
                                                     data_loading_config=data_loading_config,
                                                     pre_processor=pre_processor(config))

        environment = TrainingEnvironment.from_config(environment_config)

        trainer = KfoldTrainer(model_config=model_config, environment=environment)
        trainer.train(loader_array=loader_array, word_predictor=word_predictor, de_en_coder=de_en_coder)


def main(config_name):
    logger.info(f"Run with config '{config_name}'.")
    with TimeMeasure(enter_msg="Setup everything", exit_msg="Setup finished after {}.", writer=logger.debug):
        torch.manual_seed(0)
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

        train_loader, test_loader = get_data_loaders(meta_path=data_set_config.meta_path,
                                                     images_path=data_set_config.images_path,
                                                     transformation=transforms.Compose(transformations),
                                                     augmentation=augmentation,
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
        with TimeMeasure(enter_msg="Evaluate model:",
                         exit_msg="Evaluation finished after {}.",
                         writer=logger.debug):
            result = dict()
            for name, loader in evals:
                metrics = evaluate_model(word_prediction=word_predictor,
                                         de_en_coder=de_en_coder,
                                         model=model,
                                         data_loader=loader,
                                         device=device
                                         )
                for k in metrics.keys():
                    logger.debug(f"{name} {k}: {metrics[k]:7.4f}")
                result[name] = metrics
        return result

    trainer.model_eval = run_model_evaluation

    with TimeMeasure(enter_msg="Get trained model.",
                     exit_msg="Obtained trained model after {}.",
                     writer=logger.debug):
        if training_config.retrain:
            with TimeMeasure(enter_msg="Begin Training.",
                             exit_msg="Finished complete training after {}.",
                             writer=logger.debug):
                trainer.train(model, train_loader, device=device)
        else:
            with TimeMeasure(enter_msg="Load pre-trained model.",
                             exit_msg="Finished loading after {}.",
                             writer=logger.debug):
                trainer.load_latest_model_state_into(model)


if __name__ == "__main__":
    logger = get_htr_logger(__name__)
    logger.info("=" * 35 + " START " + "=" * 35)
    main("config_01")
    logger.info("=" * 35 + " END " + "=" * 35)
