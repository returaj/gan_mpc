"""NN model for expert."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from gan_mpc import base


class MLPCell(nn.Module):
    num_layers: int
    num_hidden_units: int
    fout: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers - 1):
            x = nn.relu(nn.Dense(self.num_hidden_units)(x))
        return nn.Dense(self.fout)(x)


class StackedMLPCell(nn.Module):
    num_layers: int
    num_hidden_units: int
    x_out: int
    u_out: int

    @nn.compact
    def __call__(self, carry, x):
        teacher_forcing, xprev = carry
        x = jnp.where(teacher_forcing, x, xprev)
        y = nn.relu(nn.Dense(self.num_hidden_units)(x))
        next_x = (
            MLPCell(self.num_layers - 1, self.num_hidden_units, self.x_out)(y)
            + x
        )
        u = nn.tanh(
            MLPCell(self.num_layers - 1, self.num_hidden_units, self.u_out)(y)
        )
        return (teacher_forcing, next_x), (next_x, u)


class LSTMCell(nn.Module):
    lstm_features: int
    num_layers: int
    num_hidden_units: int
    x_out: int
    u_out: int

    @nn.compact
    def __call__(self, carry, x):
        teacher_forcing, lstm_carry, xprev = carry
        x = jnp.where(teacher_forcing, x, xprev)
        lstm_carry, y = nn.OptimizedLSTMCell(self.lstm_features)(lstm_carry, x)
        _, (next_x, u) = StackedMLPCell(
            self.num_layers, self.num_hidden_units, self.x_out, self.u_out
        )((False, y), y)
        return (teacher_forcing, lstm_carry, next_x), (next_x, u)


class ScanMLP(nn.Module):
    num_layers: int
    num_hidden_units: int
    x_out: int
    u_out: int

    def initialize_carry(self, batch_xseq):
        return (batch_xseq[:, 0],)

    @nn.compact
    def __call__(self, carry, batch_xseq, teacher_forcing):
        """
        carry: is a tuple containing  initial value of x of size (batch_size, xdim)
        batch_xseq: (batch_size, seq, xdim)
        teacher_forcing: bool
        """

        mlp = nn.scan(
            StackedMLPCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(
            num_layers=self.num_layers,
            num_hidden_units=self.num_hidden_units,
            x_out=self.x_out,
            u_out=self.u_out,
        )

        carry = (teacher_forcing,) + carry
        _, out = mlp(carry, batch_xseq)
        return out


class ScanLSTM(ScanMLP):
    lstm_features: int

    def initialize_carry(self, batch_xseq):
        """
        batch_xseq: (batch_size, seq, xdim)
        """

        key = jax.random.PRNGKey(0)  # fix seed 0
        lstm_carry = nn.LSTMCell(self.lstm_features).initialize_carry(
            key, batch_xseq[:, 0].shape
        )
        batch_xprev = batch_xseq[:, 0]
        return (lstm_carry, batch_xprev)

    @nn.compact
    def __call__(self, carry, batch_xseq, teacher_forcing):
        """
        carry: is a tuple containing lstm_carry and
               initial value of x of size (batch_size, xdim)
        batch_xseq: (batch_size, seq, xdim)
        teacher_forcing: bool
        """

        lstm = nn.scan(
            LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(
            lstm_features=self.lstm_features,
            num_layers=self.num_layers,
            num_hidden_units=self.num_hidden_units,
            x_out=self.x_out,
            u_out=self.u_out,
        )

        carry = (teacher_forcing,) + carry
        _, out = lstm(carry, batch_xseq)
        return out


class StateAction(nn.Module, base.BaseNN):
    model: ScanMLP

    def get_init_carry(self, input):
        return self.model.initialize_carry(input)

    def get_init_params(self, seed, batch_size, seqlen, x_size):
        key = jax.random.PRNGKey(seed)
        dummy_x = jnp.zeros((batch_size, seqlen, x_size))
        return key, dummy_x

    @nn.compact
    def __call__(self, carry, batch_xseq, teacher_forcing=True):
        """
        carry: it depends if it is mlp (see ScanMLP) or lstm (see ScanLSTM) based.
        batch_xseq: (batch_size, seq, xdim)
        teacher_forcing: bool
        """

        # carry = self.get_init_carry(batch_xseq)
        return self.model(carry, batch_xseq, teacher_forcing)
