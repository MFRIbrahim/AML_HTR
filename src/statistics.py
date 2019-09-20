import os
import shutil

from util import is_directory, make_directories_for_file
from util import FrozenDict


class Statistics(object):
    """
    eons > eras > periods > epochs > ages (here batches)
    """
    __statistics_instances = dict()

    def __init__(self, name):
        self.__name = name

    def reset(self):
        path = os.path.join("statistics", self.__name)
        if is_directory(path):
            shutil.rmtree(path)

    def save_per_epoch(self, epoch, time, loss, words):
        path = os.path.join("statistics", self.__name, "1_epoch_data.txt")
        words = "\t".join(words)
        make_directories_for_file(path)
        with open(path, "a+", encoding="utf-8") as fp:
            print(f"{epoch:5d}\t{time:10d}\t{loss:14.10f}\t{words}", file=fp)

    def save_per_period(self, epoch, train_metrics, test_metrics, model=FrozenDict()):
        path = os.path.join("statistics", self.__name, "2_period_data.txt")
        make_directories_for_file(path)

        with open(path, "a+", encoding="utf-8") as fp:
            ordered_keys = sorted(list(train_metrics.keys()))
            train_data = "\t".join([f"{100 * train_metrics[key]:9.6f}" for key in ordered_keys])
            test_data = "\t".join([f"{100 * test_metrics[key]:9.6f}" for key in ordered_keys])
            model_data = "\t".join([f"{key}" for key in model])
            msg = f"{epoch:5d}\t{train_data}\t{test_data}"
            if len(model_data.strip()) > 0:
                msg += f"{model_data}\t"
            print(msg, file=fp)

    @staticmethod
    def get_instance(name):
        if name not in Statistics.__statistics_instances:
            Statistics.__statistics_instances[name] = Statistics(name)

        return Statistics.__statistics_instances[name]
