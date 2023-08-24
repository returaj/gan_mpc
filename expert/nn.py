"""NN model for expert."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from gan_mpc import base


class MLPCell(nn.Module):
    num_layers: int
    num_hidden_layers: int
    fout: int

    def __call__(self, x):
        for _ in range(self.num_layers - 1):
            x = nn.relu(nn.Dense(self.num_hidden_layers)(x))
        return nn.Dense(self.fout)(x)


class StackedMLPCell(nn.Module):
    num_layers: int
    num_hidden_units: int
    x_out: int
    u_out: int

    @nn.compact
    def __call__(self, carry, _):
        x, y = carry, carry
        y = nn.relu(nn.Dense(self.num_hidden_units)(y))
        next_x = (
            MLPCell(self.num_layers - 1, self.num_hidden_units, self.x_out)(y)
            + x
        )
        u = nn.tanh(
            MLPCell(self.num_layers - 1, self.num_hidden_units, self.u_out)(y)
        )
        return next_x, (next_x, u)


class LSTMCell(nn.Module):
    lstm_features: int
    num_layers: int
    num_hidden_units: int
    x_out: int
    u_out: int

    @nn.compact
    def __call__(self, carry, _):
        lstm_carry, x = carry
        lstm_carry, y = nn.OptimizedLSTMCell(self.lstm_features)(lstm_carry, x)
        _, (next_x, u) = StackedMLPCell(
            self.num_layers, self.num_hidden_units, self.x_out, self.u_out
        )(y, None)
        return (lstm_carry, next_x), (next_x, u)


class ScanMLP(nn.Module):
    num_layers: int
    num_hidden_units: int
    x_out: int
    u_out: int

    def initialize_carry(self, *args):
        del args
        return None

    @nn.compact
    def __call__(self, carry, x, time=1):
        del carry
        mlp = nn.scan(
            StackedMLPCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            length=time,
        )(
            num_layers=self.num_layers,
            num_hidden_units=self.num_hidden_units,
            x_out=self.x_out,
            u_out=self.u_out,
        )

        _, out = mlp(x, None)
        return out


class ScanLSTM(ScanMLP):
    lstm_features: int

    def initialize_carry(self, input_size):
        key = jax.random.PRNGKey(0)  # fix seed 0
        return nn.LSTMCell(self.lstm_features).initialize_carry(
            key, input_size
        )

    @nn.compact
    def __call__(self, carry, x, time=1):
        lstm = nn.scan(
            LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            length=time,
        )(
            lstm_features=self.lstm_features,
            num_layers=self.num_layers,
            num_hidden_units=self.num_hidden_units,
            x_out=self.x_out,
            u_out=self.u_out,
        )
        _, out = lstm((carry, x), None)
        return out


class StateAction(nn.Module, base.BaseNN):
    model: ScanMLP

    def get_init_carry(self, input_size):
        return self.model.initialize_carry(input_size)

    def get_init_params(self, seed, batch_size, x_size):
        key = jax.random.PRNGKey(seed)
        init_carry = self.get_init_carry((batch_size, x_size))
        dummy_x = jnp.zeros(x_size)
        return key, init_carry, dummy_x

    @nn.compact
    def __call__(self, carry, x):
        _, (next_x, u) = self.model(carry, x)
        return next_x, u
