"""Base learnable MPC policy."""

import functools

import jax
import jax.numpy as jnp

from gan_mpc.policy import optimizer as opt

TRAJAX_iLQR_KWARGS = {
    "maxiter": 100,
    "grad_norm_threshold": 1e-4,
    "relative_grad_norm_threshold": 0.0,
    "obj_step_threshold": 0.0,
    "inputs_step_threshold": 0.0,
    "make_psd": False,
    "psd_delta": 0.0,
    "alpha_0": 1.0,
    "alpha_min": 0.00005,
    "vjp_method": "tvlqr",
    "vjp_options": None,
}

COST_ARGS_NAME = ("goal_state",)


class BaseMPC:
    def __init__(
        self,
        config,
        cost_model,
        dynamics_model,
        expert_model,
        trajax_ilqr_kwargs=TRAJAX_iLQR_KWARGS,
    ):
        self.config = config
        self.cost_model = cost_model
        self.dynamics_model = dynamics_model
        self.expert_model = expert_model
        self.trajax_ilqr_kwargs = trajax_ilqr_kwargs
        self.solver = self.create_mpc_solver()

    def create_mpc_solver(self):
        def func(x, U, params, cost_args, dynamics_args):
            return opt.ilqr_solve(
                self.cost,
                self.dynamics,
                x,
                U,
                params,
                cost_args,
                dynamics_args,
                self.trajax_ilqr_kwargs,
            )

        return jax.jit(func)

    def __call__(self, x, params):
        init_U, goal_X = self.get_goal_states_and_init_actions(x, params)
        cost_args = (goal_X,)
        dynamics_args = ()
        return self.solver(x, init_U, params, cost_args, dynamics_args)

    def init(self, mpc_weights, cost_args, dynamics_args, expert_args):
        params = {}
        params["mpc_weights"] = jnp.array(mpc_weights)
        params["cost_params"] = self.cost_model.init(*cost_args)
        params["dynamics_params"] = self.dynamics_model.init(*dynamics_args)
        params["expert_params"] = self.expert_model.init(*expert_args)
        return params

    def cost(self, x, u, t, params, *args):
        mpc_weights = params["mpc_weights"]
        cost_params = params["cost_params"]
        return self.cost_model.get_cost(
            x, u, t, cost_params, mpc_weights, *args
        )

    def dynamics(self, x, u, t, params, *args):
        dynamics_params = params["dynamics_params"]
        return self.dynamics_model.predict(x, u, t, dynamics_params, *args)

    def get_goal_states_and_init_actions(self, x, params):
        expert_params = params["expert_params"]
        (
            init_U,
            goal_X,
        ) = self.expert_model.get_time_based_goal_states_and_init_actions(
            x, expert_params
        )
        return init_U, goal_X

    def loss(self, X, U, *args):
        raise NotImplementedError

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss_and_grad(self, X, params, loss_vmap, batch_loss_args):
        @jax.jit
        def func(x0, params, loss_args):
            init_U, goal_X = self.get_goal_states_and_init_actions(x0, params)
            cost_args = (goal_X,)
            dynamics_args = ()
            (
                high_level_loss,
                _,
                high_level_grad,
                _,
            ) = opt.bilevel_optimization(
                self.cost,
                self.dynamics,
                self.loss,
                x0,
                init_U,
                params,
                cost_args,
                dynamics_args,
                loss_args,
                self.trajax_ilqr_kwargs,
            )
            return (high_level_loss, high_level_grad)

        in_axes = (0, None) + loss_vmap
        vloss, vgrads = jax.vmap(func, in_axes=in_axes)(
            X, params, *batch_loss_args
        )
        avg_loss = jnp.mean(vloss)
        net_grads = jax.tree_map(lambda x: jnp.mean(x, axis=0), vgrads)
        return avg_loss, net_grads
