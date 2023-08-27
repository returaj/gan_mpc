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
    def get_next_state_and_action_seq(
        self, xseq, params, teacher_forcing=False
    ):
        """
        xseq: (seqlen, xdim)
        params: model's parameters
        teacher_forcing: bool
        """

        batch_xseq = jnp.expand_dims(xseq, axis=0)
        carry = self.model.get_init_carry(batch_xseq)
        batch_next_xseq, batch_useq = self.model.apply(
            params, carry, batch_xseq, teacher_forcing
        )
        # next_xseq contains xseq value at t=0. dim (seqlen, xdim+1)
        next_xseq = jnp.vstack([xseq[0], batch_next_xseq[0]])
        useq = batch_useq[0]
        return next_xseq, useq
