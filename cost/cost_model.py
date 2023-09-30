"""Cost Model for GAN-MPC."""

import functools

import jax
import jax.numpy as jnp

from gan_mpc import base


class MujocoBasedModel(base.BaseCostModel):
    def __init__(self, config, model):
        super().__init__(config)
        self.model = model

    def init(self, *args):
        model_args = self.model.get_init_params(*args)
        return self.model.init(*model_args)

    @functools.partial(jax.jit, static_argnums=0)
    def _get_staging_cost(self, xc, u, weights, goal):
        alpha = 1e-2
        # u_cost = jnp.linalg.norm(u)
        u_cost = jnp.sqrt(jnp.dot(u, u) + alpha**2) - alpha
        x_size = goal.shape[0]
        x_diff = xc[:x_size] - goal
        x_cost = jnp.sqrt(jnp.dot(x_diff, x_diff) + alpha**2) - alpha
        return jnp.array(weights) @ jnp.array([u_cost, x_cost])

    def _get_terminal_cost(self, xc, weight, params):
        return weight * self.model.get_cost(params, xc)

    @functools.partial(jax.jit, static_argnums=0)
    def get_cost(self, xc, u, t, params, weights, goal_X):
        horizon = self.config.mpc.horizon
        goal = goal_X[t]
        weights = jax.nn.sigmoid(weights)
        return jnp.where(
            t == horizon,
            self._get_terminal_cost(xc, weights[-1], params),
            self._get_staging_cost(xc, u, weights[:-1], goal),
        )
