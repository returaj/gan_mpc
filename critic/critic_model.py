"""critic model."""

from gan_mpc import base


class CriticModel(base.BaseCriticModel):
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def init(self, *args):
        model_args = self.model.get_init_params(*args)
        return self.model.init(*model_args)

    def predict(self, xseq, params):
        return self.model.apply(params, xseq)
