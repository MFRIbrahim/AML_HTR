from json import load as json_load
from types import SimpleNamespace


class Configuration(SimpleNamespace):
    def __init__(self, source):
        if type(source) == str:
            with open(source, "r") as fp:
                self.__main_config = json_load(fp)
        elif type(source) == dict:
            self.__main_config = source
        else:
            raise TypeError(f"Expected type of source to be 'str' or 'dict', but got {type(source)}.")

        super().__init__(**self.__main_config)
        self.__sub_configs = dict()

    def __call__(self, path=None):
        if path is None:
            return self.__main_config
        else:
            dictionary = self.__main_config
            for sub_dictionary in path.split("/"):
                if len(sub_dictionary) is 0:
                    continue
                dictionary = dictionary[sub_dictionary]
            return dictionary

    def __getitem__(self, path):
        if path not in self.__sub_configs:
            value = self.__call__(path=path)
            if type(value) == dict:
                self.__sub_configs[path] = Configuration(value)
            else:
                raise TypeError(f"In '{path}' a subconfig was expected, but found '{type(value)}'.")

        return self.__sub_configs[path]

    def get(self, path, default):
        try:
            return self.__call__(path)
        except KeyError:
            return default

    def if_exists(self, path, runner, default=None):
        try:
            return runner(self.__call__(path))
        except KeyError:
            return default

    def __contains__(self, item):
        return item in self.__main_config


if __name__ == "__main__":
    config = Configuration("../configs/config_01.json")
    print(config.get("prediction/hello", default="world"))
    config.if_exists("prediction", lambda d: print(d))
    config.if_exists("prediction/hello", lambda d: print(d))
    config.if_exists("prediction/char_list", lambda d: print(d))
    config.if_exists("prediction/word_prediction/eval/name", lambda d: print(d))
    print(config("prediction/char_list"))
