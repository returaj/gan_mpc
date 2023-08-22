"""L2 based learnable MPC policy."""

import jax.numpy as jnp

from gan_mpc.policy import base


class L2MPC(base.BaseMPC):
    def loss(self, X, U, desired_X):
        del U
        diff = (X - desired_X) ** 2
        return jnp.sum(jnp.mean(diff, axis=0))
