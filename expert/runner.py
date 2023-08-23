import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from gan_mpc.config import load_config
from gan_mpc.expert import nn as expert_nn
from gan_mpc.expert import trainer


def get_trainstate(model, tx, params):
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def get_model(config, state_size, action_size):
    expert_config = config.expert_prediction

    state_model = expert_nn.MLP(
        num_layers=expert_config.model.mlp.state.num_layers,
        num_hidden_units=expert_config.model.mlp.state.num_hidden_units,
        fout=state_size,
    )
    action_model = expert_nn.MLP(
        num_layers=expert_config.model.mlp.action.num_layers,
        num_hidden_units=expert_config.model.mlp.action.num_hidden_units,
        fout=action_size,
    )
    return expert_nn.StateActionNN(
        num_layers=expert_config.model.mlp.num_layers,
        num_hidden_units=expert_config.model.mlp.num_hidden_units,
        state_model=state_model,
        action_model=action_model,
    )


def get_optimizer(config):
    lr = config.expert_prediction.train.learning_rate
    return optax.adam(lr)
