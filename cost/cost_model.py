"""Cost Model for GAN-MPC."""

import jax.numpy as jnp

from gan_mpc import base


class MujocoBasedModel(base.BaseCostModel):
    def __init__(self, config, model):
        super().__init__(config)
        self.model = model

    def init(self, *args):
        model_args = self.model.get_init_params(*args)
        return self.model.init(*model_args)

    def _get_staging_cost(self, x, u, weights, goal_x):
        u_cost = jnp.linalg.norm(u)
        alpha = 0.2
        x_diff = x - goal_x
        x_cost = jnp.sqrt(jnp.dot(x_diff, x_diff) + alpha**2) - alpha
        return jnp.array(weights) @ jnp.array([u_cost, x_cost])

    def _get_terminal_cost(self, x, params, weight):
        return weight * self.model.get_cost(params, x)

    def get_cost(self, x, u, t, params, weights, goal_X):
        horizon = self.config.mpc.horizon
        goal = goal_X[t]
        return jnp.where(
            t == horizon,
            self._get_terminal_cost(x, weights[-1], params),
            self._get_staging_cost(x, u, weights[:-1], goal),
        )
