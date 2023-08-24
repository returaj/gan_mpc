"""runner code for expert prediction model."""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from gan_mpc.config import load_config
from gan_mpc.expert import nn as expert_nn
from gan_mpc.expert import trainer


def get_trainstate(model, tx, params):
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


def get_model(config, state_size, action_size):
    expert_config = config.expert_prediction


def get_optimizer(config):
    lr = config.expert_prediction.train.learning_rate
    return optax.chain(
        optax.clip_by_global_norm(max_norm=100.0), optax.adam(lr)
    )
