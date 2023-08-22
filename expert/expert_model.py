"""Expert predict model."""

import functools

import jax
import jax.numpy as jnp


class ExpertModel:
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def init(self, *args):
        model_args = self.model.get_init_params(*args)
        return self.model.init(*model_args)

    def get_goal_state_and_init_action(self, x, params):
        u, next_x = self.model.apply(params, x)
        return (u, next_x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_time_based_goal_states_and_init_actions(self, x, params, time=None):
        time = time or self.config.mpc.horizon

        def body(carry, pos):
            del pos
            x = carry
            u, next_x = self.get_goal_state_and_init_action(x, params)
            return next_x, (u, next_x)

        _, (U, next_X) = jax.lax.scan(body, x, jnp.arange(time))
        return (U, jnp.vstack((x, next_X)))
