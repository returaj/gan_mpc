"""Dynamics model for GAN-MPC."""

import jax
import jax.numpy as jnp

from gan_mpc import base


class DynamicsModel(base.BaseDynamicsModel):
    def __init__(self, config, model):
        super().__init__(config)
        self.model = model

    def init(self, *args):
        model_args = self.model.get_init_params(*args)
        return self.model.init(*model_args)

    def get_zero_carry(self, history_x):
        xsize = history_x.shape[1]
        return self.model.get_carry(jnp.zeros(xsize))

    def get_history_carry(self, history_x, history_u, params):
        """
        history_x: (history, xsize)
        history_u: (history, usize)
        """

        xsize = history_x.shape[1]
        seqlen = history_u.shape[0]

        def body(i, carry):
            x, u = history_x[i], history_u[i]
            xc = jnp.concatenate([x, carry], axis=-1)
            next_xc = self.model.apply(params, xc, u)
            carry = next_xc[xsize:]
            return carry

        # initial carry is always set to 0 vector for any x
        init_carry = self.model.get_carry(jnp.zeros(xsize))
        return jax.lax.fori_loop(0, seqlen, body, init_carry)

    def predict(self, xc, u, t, params):
        del t
        return self.model.apply(params, xc, u)
