"""Neural Network model."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from gan_mpc import base


class MLP(nn.Module, base.BaseCostNN):
    num_layers: int
    num_hidden_units: int
    fout: int

    def get_init_params(self, seed, xc_size):
        key = jax.random.PRNGKey(seed)
        dummy_xc = jnp.zeros(xc_size)
        return (key, dummy_xc)

    def get_cost(self, params, x):
        return self.apply(params, x)

    @nn.compact
    def __call__(self, xc):
        x = xc    # Can you layer norm
        # x = nn.LayerNorm(dtype=jnp.float32)(x)
        for _ in range(self.num_layers - 1):
            x = nn.relu(nn.Dense(self.num_hidden_units)(x))
        x = nn.Dense(self.fout)(x)
        return jnp.dot(x, x)
