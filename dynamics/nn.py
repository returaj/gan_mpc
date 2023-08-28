"""NN based model for dynamics."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from gan_mpc import base


class MLP(nn.Module, base.BaseDynamicsNN):
    num_layers: int
    num_hidden_units: int
    x_out: int

    def get_carry(self, x):
        shape = x.shape[:-1]
        return jnp.empty(shape=(*shape, 0))

    def get_init_params(self, seed, u_size):
        key = jax.random.PRNGKey(seed)
        dummy_x = jnp.zeros(self.x_out)
        dummy_carry = self.get_carry(dummy_x)
        dummy_xc = jnp.concatenate([dummy_x, dummy_carry], axis=-1)
        dummy_u = jnp.zeros(u_size)
        return (key, dummy_xc, dummy_u)

    @nn.compact
    def __call__(self, xc, u):
        x, carry = jnp.array_split(xc, [self.x_out], axis=-1)
        q = jnp.concatenate([x, u], axis=-1)
        for _ in range(self.num_layers - 1):
            q = nn.relu(nn.Dense(self.num_hidden_units)(q))
        next_x = nn.Dense(self.x_out)(q) + x
        return jnp.concatenate([next_x, carry], axis=-1)


class LSTM(MLP):
    lstm_features: int

    def get_carry(self, x):
        key = jax.random.PRNGKey(0)  # fix key
        carry = nn.OptimizedLSTMCell(
            self.lstm_features, parent=None
        ).initialize_carry(key, input_shape=x.shape)
        return jnp.concatenate(carry, axis=-1)

    @nn.compact
    def __call__(self, xc, u):
        x, *carry = jnp.array_split(
            xc, [self.x_out, self.x_out + self.lstm_features], axis=-1
        )
        q = jnp.concatenate([x, u], axis=-1)
        carry, q = nn.OptimizedLSTMCell(self.lstm_features)(carry, q)
        for _ in range(self.num_layers - 1):
            q = nn.relu(nn.Dense(self.num_hidden_units)(q))
        next_x = nn.Dense(self.x_out)(q) + x
        return jnp.concatenate([next_x, *carry], axis=-1)
