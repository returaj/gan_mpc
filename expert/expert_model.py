"""Expert predict model."""

import functools

import jax
import jax.numpy as jnp


class ExpertModel:
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def init(self, *args):
        model_args = self.model.get_init_params(*args)
        return self.model.init(*model_args)

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_next_state_and_action_seq(self, x, params, time=None):
        time = time or self.config.mpc.horizon
        carry = self.model.get_init_carry(x.shape)
        _, (next_X, U) = self.model.apply(params, carry, x)
        return (jnp.vstack((x, next_X)), U)
