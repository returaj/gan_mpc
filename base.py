"""Base file for GAN-MPC."""


class BaseCostModel:
    def __init__(self, config):
        self.config = config

    def init(self, *args):
        raise NotImplementedError

    def get_cost(x, u, t, *cost_args):
        raise NotImplementedError


class BaseDynamicsModel:
    def __init__(self, config):
        self.config = config

    def init(self, *args):
        raise NotImplementedError

    def predict(x, u, t, *dynamics_args):
        raise NotImplementedError


class BaseNN:
    def get_init_params(self, *args):
        raise NotImplementedError


class BaseCostNN(BaseNN):
    def get_cost(self, *args):
        raise NotImplementedError
