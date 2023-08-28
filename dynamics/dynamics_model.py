"""Dynamics model for GAN-MPC."""

from gan_mpc import base


class DynamicsModel(base.BaseDynamicsModel):
    def __init__(self, config, model):
        super().__init__(config)
        self.model = model

    def init(self, *args):
        model_args = self.model.get_init_params(*args)
        return self.model.init(*model_args)

    def get_carry(self, x):
        return self.model.get_carry(x)

    def predict(self, x, u, t, params):
        del t
        return self.model.apply(params, x, u)
