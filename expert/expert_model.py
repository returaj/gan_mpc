"""Expert predict model."""

import functools

import jax
import jax.numpy as jnp

from gan_mpc import utils
from gan_mpc.expert import nn as expert_nn


class ExpertModel:
    def __init__(self, config, model):
        self.config = config
        self.model = model

    @staticmethod
    def get_model(model_config, x_size, u_size):
        if model_config.use == "lstm":
            lstm_config = model_config.lstm
            model = expert_nn.ScanLSTM(
                lstm_features=lstm_config.lstm_features,
                num_layers=lstm_config.num_layers,
                num_hidden_units=lstm_config.num_hidden_units,
                x_out=x_size,
                u_out=u_size,
            )
        elif model_config.use == "mlp":
            mlp_config = model_config.mlp
            model = expert_nn.ScanMLP(
                num_layers=mlp_config.num_layers,
                num_hidden_units=mlp_config.num_hidden_units,
                x_out=x_size,
                u_out=u_size,
            )
        else:
            raise ValueError("Choose either mlp or lstm model.")
        return expert_nn.StateAction(model)

    def init(self, load_params, *args):
        config = self.config
        if load_params:
            env_type, env_name = config.env.type, config.env.expert.name
            env_id = config.mpc.model.expert.load_id
            params_path = f"trained_models/expert/{env_type}/{env_name}/{env_id}/params.npy"
            params = utils.load_params(params_path)
        else:
            model_args = self.model.get_init_params(*args)
            params = self.model.init(*model_args)
        return params

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_zero_carry(self, xseq):
        batch_xseq = jnp.expand_dims(xseq, axis=0)
        batch_carry = self.model.get_init_carry(batch_xseq)
        carry = jax.tree_map(
            lambda x: jnp.squeeze(batch_carry, axis=0), batch_carry
        )
        return carry

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_history_carry(self, history_x, params):
        history, x = history_x[:-1], history_x[-1]
        init_carry = self.get_zero_carry(history)
        carry, _ = self.get_carry_next_state_and_action_seq(
            init_carry, history, params, teacher_forcing=True
        )
        *model_carry, _ = carry
        return tuple(model_carry) + (x,)

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_carry_next_state_and_action_seq(
        self, carry, xseq, params, teacher_forcing=False
    ):
        """
        xseq: (seqlen, xdim)
        params: model's parameters
        teacher_forcing: bool
        """

        batch_xseq = jnp.expand_dims(xseq, axis=0)
        batch_carry = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), carry)
        batch_carry, (batch_next_xseq, batch_useq) = self.model.apply(
            params, batch_carry, batch_xseq, teacher_forcing
        )
        # next_xseq contains xseq value at t=0. dim (seqlen, xdim+1)
        next_xseq = jnp.vstack([xseq[0], batch_next_xseq[0]])
        useq = batch_useq[0]
        carry = jax.tree_map(
            lambda x: jnp.squeeze(batch_carry, axis=0), batch_carry
        )
        return carry, (next_xseq, useq)
