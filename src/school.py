from copy import copy
from os.path import join as p_join

import Levenshtein
import numpy as np
import torch
from torch.nn import CTCLoss, functional as F
from torch.optim import Adam

from statistics import Statistics
from model import get_model_by_name
from transformations import right_strip, word_tensor_to_list
from util import TimeMeasure, save_checkpoint, load_latest_checkpoint, FrozenDict, get_htr_logger

logger = get_htr_logger(__name__)


class TrainingEnvironment(object):
    def __init__(self, max_epochs=20,
                 warm_start=False,
                 loss_name="CTC",
                 optimizer_name="Adam",
                 optimizer_args=FrozenDict(),
                 save_interval=10):
        self.__max_epochs = max_epochs
        self.__warm_start = warm_start
        self.__save_interval = save_interval
        self.__loss_function = loss_function_by_name(loss_name)
        self.__loss_name = loss_name
        self.__optimizer_creator = optimizer_creator_by_name(optimizer_name)
        self.__optimizer_args = optimizer_args
        self.__optimizer_name = optimizer_name

    @property
    def max_epochs(self):
        return self.__max_epochs

    @property
    def warm_start(self):
        return self.__warm_start

    @property
    def loss_function(self):
        return self.__loss_function

    @property
    def loss_name(self):
        return self.__loss_name

    def create_optimizer(self, model, learning_rate):
        return self.__optimizer_creator(model, learning_rate, **self.__optimizer_args)

    @property
    def optimizer_name(self):
        return self.__optimizer_name

    @property
    def save_interval(self):
        return self.__save_interval

    def to_dict(self):
        return {"max_epochs": self.max_epochs,
                "warm_start": self.warm_start,
                "loss": self.loss_name,
                "optimizer": self.optimizer_name,
                "optimizer_args": self.__optimizer_args,
                "save_interval": self.save_interval}

    @staticmethod
    def from_dict(dictionary):
        return TrainingEnvironment(max_epochs=dictionary["max_epochs"],
                                   warm_start=dictionary["warm_start"],
                                   loss_name=dictionary["loss"],
                                   optimizer_name=dictionary["optimizer"],
                                   optimizer_args=dictionary["optimizer_args"],
                                   save_interval=dictionary["save_interval"]
                                   )

    @staticmethod
    def from_config(environment_config):
        return TrainingEnvironment(max_epochs=environment_config.epochs,
                                   warm_start=environment_config.warm_start,
                                   loss_name=environment_config("loss/name"),
                                   optimizer_name=environment_config("optimizer/name"),
                                   optimizer_args=environment_config("optimizer/parameters")
                                   )


def loss_function_by_name(name):
    if name == "CTC":
        return CTCLoss()
    else:
        raise RuntimeError("Unknown loss function '{}'".format(name))


def optimizer_creator_by_name(name):
    if name == "Adam":
        def creator(model, learning_rate, **kwargs):
            return Adam(model.parameters(), lr=learning_rate, **kwargs)

        return creator
    else:
        raise RuntimeError("Unknown optimizer '{}'".format(name))


class Trainer(object):
    def __init__(self,
                 name,
                 model,
                 word_prediction,
                 dynamic_learning_rate=lambda idx: 1e-4,
                 print_enabled=True,
                 environment=None):
        self.__name = name
        self.__model = model
        self.__word_prediction = word_prediction
        self.__learning_rate_adaptor = dynamic_learning_rate
        self.__print_enabled = print_enabled
        self.__environment = TrainingEnvironment() if environment is None else environment
        self.model_eval = lambda current_model: dict()

    def train(self, train_loader, current_epoch=0, device="cpu"):
        logger.info("Enter training mode.")
        total_epochs = current_epoch
        last_save, loss = 0, None
        stats = Statistics.get_instance(self.__name)

        logger.info(f"Try warm start? - {'Yes' if self.__environment.warm_start else 'No'}")
        if self.__environment.warm_start:
            try:
                total_epochs, state_dict, loss = self.__load_progress()
                self.__model.load_state_dict(state_dict)
            except RuntimeError:
                logger.warning("Warm start was not possible!")

        for epoch_idx in range(1, self.__environment.max_epochs + 1):
            enter_msg = f"Train Epoch: {epoch_idx: 4d} (total: {total_epochs + 1: 4d})"
            with TimeMeasure(enter_msg=enter_msg,
                             writer=logger.info,
                             print_enabled=self.__print_enabled) as tm:
                current_learning_rate = self.__learning_rate_adaptor(total_epochs)
                loss, words = self.core_training(train_loader, current_learning_rate, device)
                logger.info("loss: {}".format(loss))
                total_epochs += 1

                stats.save_per_epoch(total_epochs, tm.delta, loss, words)
                if epoch_idx % self.__environment.save_interval is 0:
                    last_save = total_epochs
                    self.__save_progress(total_epochs, self.__model, loss)
                    self.__save_period_stats(total_epochs)

        if last_save < total_epochs:
            logger.info("final save")
            self.__save_progress(total_epochs, self.__model, loss)
            self.__save_period_stats(total_epochs)

        return self.__model

    def __load_progress(self):
        directory = p_join("trained_models", self.__name)
        dictionary = load_latest_checkpoint(directory)
        return dictionary["total_epochs"], dictionary["model_states"], dictionary["loss"]

    def core_training(self, train_loader, learning_rate, device):
        loss_fct = self.__environment.loss_function.to(device)
        optimizer = self.__environment.create_optimizer(self.__model, learning_rate)
        self.__model.train(mode=True)
        mean_loss = 0
        first_batch_words = list()

        for (batch_id, (feature_batch, label_batch)) in enumerate(train_loader):
            self.__model.init_hidden(batch_size=feature_batch.size()[0], device=device)
            feature_batch = feature_batch.to(device)
            label_batch = [np.asarray(right_strip(list(map(int, word)), 1)) for word in
                           word_tensor_to_list(label_batch)]
            optimizer.zero_grad()
            model_out = self.__model(feature_batch)
            ctc_input = F.log_softmax(model_out, dim=-1).to(device)
            input_lengths = torch.full(size=(len(feature_batch),),
                                       fill_value=model_out.shape[0],
                                       dtype=torch.long
                                       ).to(device)
            ctc_target = np.concatenate(label_batch, axis=0)  # TODO: Check axis
            target_lengths = [len(w) for w in label_batch]
            target_lengths = torch.Tensor(target_lengths).to(device).type(torch.long)
            ctc_target = torch.Tensor(ctc_target).to(device).type(torch.long)

            if batch_id == 0:
                first_batch_words = self.__print_words_in_batch(ctc_input)

            loss = loss_fct(ctc_input, ctc_target, input_lengths, target_lengths)
            mean_loss += loss.item()
            loss.backward()
            optimizer.step()

        return mean_loss / len(train_loader), first_batch_words

    def __print_words_in_batch(self, ctc_input):
        cpu_input = np.array(copy(ctc_input).detach().cpu())
        out = self.__word_prediction(cpu_input)
        # for i, word in enumerate(out):
        #   logger.debug("{:02d}: '{}'".format(i, word))
        return out

    def __save_progress(self, total_epochs, model, loss):
        with TimeMeasure(enter_msg="Saving progress...",
                         writer=logger.debug,
                         print_enabled=self.__print_enabled):
            path = p_join("trained_models", self.__name, "epoch-{:05d}.pt".format(total_epochs))
            save_checkpoint(path, total_epochs, model, loss, self.__environment)

    def __save_period_stats(self, total_epochs):
        stats = Statistics.get_instance(self.__name)
        accs = self.model_eval()

        stats.save_per_period(total_epochs,
                              train_acc=accs.get("train", 0.0),
                              test_acc=accs.get("test", 0.0))

    def load_latest_model(self):
        total_epochs, state_dict, loss = self.__load_progress()
        self.__model.load_state_dict(state_dict)
        return self.__model


def evaluate_model(de_en_coder, word_prediction, model, data_loader, device):
    correct, counter = 0, 0
    character_error_rate, word_error_rate = 0, 0

    with torch.no_grad():
        for batch_idx, (feature_batch, label_batch) in enumerate(data_loader):
            feature_batch = feature_batch.to(device)
            label_batch = [right_strip(word, 1.0) for word in word_tensor_to_list(label_batch)]
            label_batch = [de_en_coder.decode_word(word) for word in label_batch]
            model.init_hidden(batch_size=feature_batch.size()[0], device=device)

            output = F.softmax(model(feature_batch), dim=-1)
            output = np.array(output.cpu())
            prediction = word_prediction(output)

            for i in range(len(prediction)):
                counter += 1

                token, target = prediction[i], label_batch[i]

                character_error_rate += Levenshtein.distance(token, target) / len(target)
                if token == target:
                    correct += 1

                single_words_pred, single_words_target = prediction[i].split(" "), label_batch[i].split(" ")

                word_error_rate += __calculate_word_error_rate(single_words_pred, single_words_target)

    return {"Accuracy": correct / counter,
            "Character Error Rate": character_error_rate / counter,
            "Word Error Rate": word_error_rate / counter}


def __calculate_word_error_rate(single_words_pred, single_words_target):
    word_count = len(single_words_target)
    padded_words = [""] * abs(len(single_words_pred) - len(single_words_target))

    if len(single_words_pred) > len(single_words_target):
        single_words_target.extend(padded_words)

    elif len(single_words_pred) < len(single_words_target):
        single_words_pred.extend(padded_words)

    errors = sum([1 for j in range(len(single_words_pred)) if single_words_pred[j] != single_words_target[j]])

    return errors / word_count


class KfoldTrainer(object):
    def __init__(self,
                 name,
                 model_config,
                 word_prediction,
                 dynamic_learning_rate=lambda idx: 1e-4,
                 environment=None,):
        self.__name = name
        self.__model_config = model_config
        self.__word_prediction = word_prediction
        self.__learning_rate_adaptor = dynamic_learning_rate
        self.__environment = TrainingEnvironment() if environment is None else environment
        self.__stats = Statistics.get_instance(self.__name)

    def train(self, loader_array, word_predictor, de_en_coder, current_epoch=0, device="cpu"):
        logger.info("Enter training mode.")
        model_id = 0
        for loaders in loader_array:
            model = get_model_by_name(self.__model_config.name)(self.__model_config.parameters).to(device)
            total_epochs = current_epoch
            self.train_single_model(model, loaders, total_epochs, device, de_en_coder, model_id)
            model_id += 1

    def train_single_model(self, model, loaders, total_epochs, device, de_en_coder, model_id):
        train_loader = loaders[0]
        train_eval_loader = loaders[1]
        test_loader = loaders[2]
        for epoch_idx in range(1, self.__environment.max_epochs + 1):
            enter_msg = f"Train Epoch: {epoch_idx: 4d} (total: {total_epochs + 1: 4d})"
            with TimeMeasure(enter_msg=enter_msg,
                             writer=logger.info,
                             print_enabled=True) as tm:
                current_learning_rate = self.__learning_rate_adaptor(total_epochs)
                loss, words = self.core_training(model, train_loader, current_learning_rate, device)
                logger.info(f"loss: {loss}")
                total_epochs += 1
                if epoch_idx % self.__environment.save_interval is 0:
                    train_metrics = evaluate_model(de_en_coder=de_en_coder,
                                                   word_prediction=self.__word_prediction,
                                                   model=model,
                                                   data_loader=train_eval_loader,
                                                   device=device)
                    test_metrics = evaluate_model(de_en_coder=de_en_coder,
                                                  word_prediction=self.__word_prediction,
                                                  model=model,
                                                  data_loader=test_loader,
                                                  device=device)
                    model_data = {"name": f"{model.__class__.__name__}_{model_id:03d}"}
                    self.__stats.save_per_period(total_epochs, train_metrics, test_metrics, model_data)

    def core_training(self, model, train_loader, learning_rate, device):
        loss_fct = self.__environment.loss_function.to(device)
        optimizer = self.__environment.create_optimizer(model, learning_rate)
        model.train(mode=True)
        mean_loss = 0
        first_batch_words = list()

        for (batch_id, (feature_batch, label_batch)) in enumerate(train_loader):
            if batch_id % (len(train_loader) / 100) == 0:
                logger.debug(f"Batch: {batch_id:04d}")

            model.init_hidden(batch_size=feature_batch.size()[0], device=device)
            feature_batch = feature_batch.to(device)
            label_batch = [np.asarray(right_strip(list(map(int, word)), 1)) for word in
                           word_tensor_to_list(label_batch)]
            optimizer.zero_grad()

            model_out = model(feature_batch)
            ctc_input = F.log_softmax(model_out, dim=-1).to(device)
            input_lengths = torch.full(size=(len(feature_batch),),
                                       fill_value=model_out.shape[0],
                                       dtype=torch.long
                                       ).to(device)
            ctc_target = np.concatenate(label_batch, axis=0)  # TODO: Check axis
            target_lengths = [len(w) for w in label_batch]
            target_lengths = torch.Tensor(target_lengths).to(device).type(torch.long)
            ctc_target = torch.Tensor(ctc_target).to(device).type(torch.long)

            if batch_id == 0:
                first_batch_words = self.__print_words_in_batch(ctc_input)

            loss = loss_fct(ctc_input, ctc_target, input_lengths, target_lengths)
            mean_loss += loss.item()
            loss.backward()
            optimizer.step()

        return mean_loss / len(train_loader), first_batch_words

    def __print_words_in_batch(self, ctc_input):
        cpu_input = np.array(copy(ctc_input).detach().cpu())
        out = self.__word_prediction(cpu_input)
        for i, word in enumerate(out):
           logger.debug("{:02d}: '{}'".format(i, word))
        return out
