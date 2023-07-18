"""Dynamics model for GAN-MPC."""

import jax.numpy as jnp

from gan_mpc import base


class DynamicsModel(base.BaseDynamicsModel):
  def __init__(self, config, model):
    super().__init__(config)
    self.model = model

  def init(self, *args):
    model_args = self.model.get_init_params(*args)
    return self.model.init(*model_args)

  def predict(self, x, u, t, params):
    return self.model.apply(params, x, u)
