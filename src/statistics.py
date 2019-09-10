import os
import shutil


class Statistics(object):
    """
    eons > eras > periods > epochs > ages (here batches)
    """
    __statistics_instances = dict()

    def __init__(self, name):
        self.__name = name

    def reset(self):
        path = os.path.join("statistics", self.__name)
        shutil.rmtree(path)

    def save_per_epoch(self, epoch, time, loss, words):
        path = os.path.join("statistics", self.__name, "1_epoch_data.txt")
        words = "\t".join(words)
        with open(path, "a+", encoding="utf-8") as fp:
            print(f"{epoch:5d}\t{time:10d}\t{loss:14.10f}\t{words}", file=fp)

    def save_per_period(self, epoch, train_acc, test_acc):
        path = os.path.join("statistics", self.__name, "2_period_data.txt")
        with open(path, "a+", encoding="utf-8") as fp:
            print(f"{epoch:5d}\t{100*train_acc:9.6f}\t{100*test_acc:9.6f}\t", file=fp)

    @staticmethod
    def get_instance(name):
        if name not in Statistics.__statistics_instances:
            Statistics.__statistics_instances[name] = Statistics(name)

        return Statistics.__statistics_instances[name]
