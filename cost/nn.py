"""Neural Network model."""

import jax
import jax.numpy as jnp
import flax.linen as nn

from gan_mpc import base


class MLP(nn.Module, base.BaseCostNN):
  num_layers: int
  num_hidden_units: int
  fout: int

  def get_init_params(self, seed, x_size):
    key = jax.random.PRNGKey(seed)
    dummy_x = jnp.zeros(x_size)
    return (key, dummy_x)

  def get_cost(self, params, x):
    return self.apply(params, x)

  @nn.compact
  def __call__(self, x):
    for _ in range(self.num_layers-1):
      x = nn.relu(nn.Dense(self.num_hidden_units)(x))
    x = nn.Dense(self.fout)(x)
    return jnp.dot(x, x)
