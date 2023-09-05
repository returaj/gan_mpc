"""L2 based learnable MPC policy."""

import functools

import jax
import jax.numpy as jnp

from gan_mpc.policy import base


class L2MPC(base.BaseMPC):
    @functools.partial(jax.jit, static_argnums=(0,))
    def loss(self, XC, U, desired_X):
        del U
        x_size = desired_X.shape[-1]
        X, _ = jnp.split(XC, [x_size], axis=-1)
        diff = (X - desired_X) ** 2
        return jnp.sum(jnp.mean(diff, axis=0))
