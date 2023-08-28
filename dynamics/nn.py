"""NN based model for dynamics."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from gan_mpc import base


class MLP(nn.Module, base.BaseNN):
    num_layers: int
    num_hidden_units: int
    fout: int

    def get_init_params(self, seed, x_size, u_size):
        key = jax.random.PRNGKey(seed)
        dummy_x = jnp.zeros(x_size)
        dummy_u = jnp.zeros(u_size)
        return (key, dummy_x, dummy_u)

    @nn.compact
    def __call__(self, x, u):
        q = jnp.concatenate([x, u], axis=-1)
        for _ in range(self.num_layers - 1):
            q = nn.relu(nn.Dense(self.num_hidden_units)(q))
        q = nn.Dense(self.fout)(q)
        return x + q


class LSTM(nn.Module, base.BaseNN):
    lstm_features: int
    num_hidden_units: int
