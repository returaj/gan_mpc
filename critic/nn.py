"""critic flax model."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from gan_mpc import base


class LSTM(nn.Module, base.BaseNN):
    lstm_features: int
    num_layers: int
    num_hidden_units: int
    fout: int = 1

    def get_carry(self, xseq):
        key = jax.random.PRNGKey(0)
        return nn.OptimizedLSTMCell(
            self.lstm_features, parent=None
        ).initialize_carry(key, input_shape=(xseq.shape[1],))

    def get_init_params(self, seed, xsize):
        key = jax.random.PRNGKey(seed)
        dummy_xseq = jnp.zeros((1, xsize))
        return key, dummy_xseq

    @nn.compact
    def __call__(self, xseq):
        lstm = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )(features=self.lstm_features)

        init_carry = self.get_carry(xseq)
        (_, out), _ = lstm(init_carry, xseq)

        for _ in range(self.num_layers - 1):
            out = nn.relu(nn.Dense(self.num_hidden_units)(out))
        return nn.Dense(self.fout)(out)
