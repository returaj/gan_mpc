"""L2 based learnable MPC policy."""

import functools

import jax
import jax.numpy as jnp

from gan_mpc.policy import base


class L2MPC(base.BaseMPC):
    @functools.partial(jax.jit, static_argnums=(0,))
    def loss(self, xcseq, useq, params, desired_xseq):
        del useq, params
        x_size = desired_xseq.shape[-1]
        xseq, _ = jnp.split(xcseq, [x_size], axis=-1)
        diff = (xseq - desired_xseq) ** 2
        return jnp.sum(jnp.mean(diff, axis=0))
