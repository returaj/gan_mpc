"""Config class to load hyperparameters.yaml"""

import yaml


class Config:
    @staticmethod
    def from_yaml(filepath):
        """
        reads configurations from yaml file
        """
        with open(filepath, "r") as fp:
            dataMap = yaml.safe_load(fp)
        return Config.from_dict(dataMap)

    @staticmethod
    def from_dict(dataMap):
        """
        reads configurations from dictonary

        For example:
        params = {"name": "CNN", "type": "Neural Network", "value": {"a": 1, "b": 2}}
        config = Config.from_dict(params)
        # then one can access the config parameters as
        print(config.name) # prints: CNN
        print(config.type) # prints: Neural Network
        print(config.value.a) # prints: 1
        """
        config = Config()
        for name, value in dataMap.items():
            if isinstance(value, dict):
                value = Config.from_dict(value)
            setattr(config, name, value)
        return config

    def to_dict(self):
        ret = {}
        attribute_dict = self.__dict__
        for k, v in attribute_dict.items():
            if isinstance(v, Config):
                v = v.to_dict()
            ret[k] = v
        return ret
