import time


class TimeMeasure(object):
    def __init__(self, enter_msg, exit_msg="{} ms.", writer=print):
        self.__enter_msg = enter_msg
        self.__exit_msg = exit_msg
        self.__writer = writer
        self.__time = None

    def __enter__(self):
        self.__start = time.time()
        self.__writer(self.__enter_msg)

    def __exit__(self, exc_type, exc_val, exc_tb):
        delta = time.time() - self.__start
        delta = int(delta * 1000)
        self.__writer(self.__exit_msg.format(delta))
