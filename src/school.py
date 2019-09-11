from copy import copy
from os.path import join as p_join
from torch.optim import Adam
from torch.nn import CTCLoss, functional as F
import numpy as np
import torch
import Levenshtein

from statistics import Statistics
from transformations import rstrip, word_tensor_to_list
from util import TimeMeasure, save_checkpoint, load_latest_checkpoint


class TrainingEnvironment(object):
    def __init__(self, max_epochs=20, warm_start=False, loss_name="CTC", optimizer_name="Adam", optimizer_args={},
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
    def __init__(self, name, word_prediction, dynamic_learning_rate=lambda idx: 1e-4, print_enabled=True, writer=print,
                 environment=None):
        self.__name = name
        self.__word_prediction = word_prediction
        self.__learning_rate_adaptor = dynamic_learning_rate
        self.__print_enabled = print_enabled
        self.__writer = writer
        self.__environment = TrainingEnvironment() if environment is None else environment
        self.model_eval = lambda: dict()

    def train(self, model, train_loader, current_epoch=0, device="cpu"):
        total_epochs = current_epoch
        last_save, loss = 0, None
        stats = Statistics.get_instance(self.__name)

        if self.__environment.warm_start:
            try:
                total_epochs, state_dict, loss = self.__load_progress()
                model.load_state_dict(state_dict)
            except RuntimeError:
                self.__writer("Warm start was not possible!")

        for epoch_idx in range(1, self.__environment.max_epochs + 1):
            enter_msg = "Train Epoch: {: 4d} (total: {: 4d})".format(epoch_idx, total_epochs + 1)
            with TimeMeasure(enter_msg=enter_msg, writer=self.__writer, print_enabled=self.__print_enabled) as tm:
                current_learning_rate = self.__learning_rate_adaptor(total_epochs)
                loss, words = self.core_training(model, train_loader, current_learning_rate, device)
                self.__writer("loss: {}".format(loss))
                total_epochs += 1

                stats.save_per_epoch(total_epochs, tm.delta, loss, words)
                if epoch_idx % self.__environment.save_interval is 0:
                    last_save = total_epochs
                    self.__save_progress(total_epochs, model, loss)
                    self.__save_period_stats(total_epochs)

        if last_save < total_epochs:
            print("final save")
            self.__save_progress(total_epochs, model, loss)
            self.__save_period_stats(total_epochs)

    def __load_progress(self):
        directory = p_join("trained_models", self.__name)
        dictionary = load_latest_checkpoint(directory)
        return dictionary["total_epochs"], dictionary["model_states"], dictionary["loss"]

    def core_training(self, model, train_loader, learning_rate, device):
        loss_fct = self.__environment.loss_function.to(device)
        optimizer = self.__environment.create_optimizer(model, learning_rate)
        model.train(mode=True)
        mean_loss = 0
        first_batch_words = list()

        for (batch_id, (feature_batch, label_batch)) in enumerate(train_loader):
            model.init_hidden(batch_size=feature_batch.size()[0], device=device)
            feature_batch = feature_batch.to(device)
            label_batch = [np.asarray(rstrip(list(map(int, word)), 1)) for word in word_tensor_to_list(label_batch)]

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
            print("{:02d}: '{}'".format(i, word))
        return out

    def __save_progress(self, total_epochs, model, loss):
        with TimeMeasure(enter_msg="Saving progress...", writer=self.__writer,
                         print_enabled=self.__print_enabled):
            path = p_join("trained_models", self.__name, "epoch-{:05d}.pt".format(total_epochs))
            save_checkpoint(path, total_epochs, model, loss, self.__environment)

    def __save_period_stats(self, total_epochs):
        stats = Statistics.get_instance(self.__name)
        accs = self.model_eval()

        stats.save_per_period(total_epochs,
                              train_acc=accs.get("train", 0.0),
                              test_acc=accs.get("test", 0.0)
                              )

    def load_latest_model_state_into(self, model):
        total_epochs, state_dict, loss = self.__load_progress()
        model.load_state_dict(state_dict)


def evaluate_model(de_en_coder, word_prediction, model, data_loader, device):
    correct, counter, WER, CER = 0, 0, 0, 0

    with torch.no_grad():
        for batch_idx, (feature_batch, label_batch) in enumerate(data_loader):
            feature_batch = feature_batch.to(device)
            label_batch = [rstrip(word, 1.0) for word in word_tensor_to_list(label_batch)]
            label_batch = [de_en_coder.decode_word(word) for word in label_batch]
            model.init_hidden(batch_size=feature_batch.size()[0], device=device)

            output = F.softmax(model(feature_batch), dim=-1)
            output = np.array(output.cpu())
            prediction = word_prediction(output)

            for i in range(len(prediction)):
                errors = 0
                counter += 1

                token = prediction[i]
                target = label_batch[i]

                CER += Levenshtein.distance(token, target)/len(target)
                if token == target:
                    correct += 1

                single_words_pred = prediction[i].split(" ")
                single_words_target = label_batch[i].split(" ")

                # pad array if more or less words have been predicted than acutally existed
                if len(single_words_pred) > len(single_words_target):
                    single_words_target.extend([""]*(len(single_words_pred) - len(single_words_target)))
                if len(single_words_pred) < len(single_words_target):
                    single_words_pred.extend([""]*(len(single_words_target) - len(single_words_pred)))
                for j in range(len(single_words_pred)):
                    if single_words_pred[j] != single_words_target[j]:
                        errors += 1
                WER += errors/len(label_batch[i].split(" "))
                return {"CER": CER/counter, "WER": WER/counter, "Accuracy": correct/counter}
