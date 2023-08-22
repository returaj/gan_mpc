"""NN model for expert."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from gan_mpc import base


class MLP(nn.Module):
    num_layers: int
    num_hidden_units: int
    fout: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.relu(nn.Dense(self.num_hidden_units)(x))
        return nn.Dense(self.fout)(x)


class StateActionNN(MLP, base.BaseNN):
    num_layers: int
    num_hidden_units: int
    action_model: MLP
    state_model: MLP

    def get_init_params(self, seed, x_size):
        key = jax.random.PRNGKey(seed)
        dummy_x = jnp.zeros(x_size)
        return (key, dummy_x)

    @nn.compact
    def __call__(self, x):
        q = x
        for _ in range(self.num_layers):
            q = nn.relu(nn.Dense(self.num_hidden_units)(q))
        u = self.action_model(q)
        next_x = x + self.state_model(q)
        return (u, next_x)
