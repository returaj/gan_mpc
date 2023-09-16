"""Base learnable MPC policy."""

import functools

import jax
import jax.numpy as jnp

from gan_mpc.policy import optimizers as opt

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
}

COST_ARGS_NAME = ("goal_state",)


class BaseMPC:
    def __init__(
        self,
        config,
        cost_model,
        dynamics_model,
        expert_model,
        loss_vmap=(0,),
        trajax_ilqr_kwargs=TRAJAX_iLQR_KWARGS,
    ):
        self.config = config
        self.cost_model = cost_model
        self.dynamics_model = dynamics_model
        self.expert_model = expert_model
        self.loss_vmap = loss_vmap
        self.trajax_ilqr_kwargs = trajax_ilqr_kwargs
        self.solver = self.create_mpc_solver()

    def create_mpc_solver(self):
        def func(xc, U, params, cost_args, dynamics_args):
            return opt.ilqr_solve(
                self.cost,
                self.dynamics,
                xc,
                U,
                params,
                cost_args,
                dynamics_args,
                self.trajax_ilqr_kwargs,
            )

        return jax.jit(func)

    def __call__(self, x, params):
        (
            goal_xseq,
            init_useq,
            init_carry,
        ) = self.get_goal_states_init_actions_and_carry(x, params)
        cost_args = (goal_xseq,)
        dynamics_args = ()
        xc = jnp.concatenate([x, init_carry], axis=-1)
        return self.solver(xc, init_useq, params, cost_args, dynamics_args)

    @functools.partial(jax.jit, static_argnums=0)
    def get_optimal_action(self, x, params):
        _, useq, *_ = self(x, params)
        return useq[0]

    def init(self, mpc_weights, cost_args, dynamics_args, expert_args):
        params = {}
        params["mpc_weights"] = jnp.array(mpc_weights, dtype=jnp.float32)
        params["cost_params"] = self.cost_model.init(*cost_args)
        params["dynamics_params"] = self.dynamics_model.init(*dynamics_args)
        params["expert_params"] = self.expert_model.init(*expert_args)
        return params

    def cost(self, xc, u, t, params, *args):
        mpc_weights = params["mpc_weights"]
        cost_params = params["cost_params"]
        return self.cost_model.get_cost(
            xc, u, t, cost_params, mpc_weights, *args
        )

    def dynamics(self, xc, u, t, params, *args):
        dynamics_params = params["dynamics_params"]
        return self.dynamics_model.predict(xc, u, t, dynamics_params, *args)

    def get_carry(self, x):
        return self.dynamics_model.get_carry(x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_goal_states_init_actions_and_carry(self, x, params):
        expert_params = params["expert_params"]
        xseq = jnp.vstack(
            [x, jnp.zeros((self.config.mpc.horizon - 1, x.shape[0]))]
        )
        (goal_X, init_U) = self.expert_model.get_next_state_and_action_seq(
            xseq, expert_params
        )
        init_carry = self.get_carry(x)
        return goal_X, init_U, init_carry

    def loss(self, xcseq, useq, params, *args):
        raise NotImplementedError

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss_and_grad(self, X, params, batch_loss_args):
        @jax.jit
        def func(x, params, *loss_args):
            (
                goal_xseq,
                init_useq,
                init_carry,
            ) = self.get_goal_states_init_actions_and_carry(x, params)
            cost_args = (goal_xseq,)
            dynamics_args = ()
            xc = jnp.concatenate([x, init_carry], axis=-1)
            (
                high_level_loss,
                _,
                high_level_grad,
                _,
            ) = opt.bilevel_optimization(
                self.cost,
                self.dynamics,
                self.loss,
                xc,
                init_useq,
                params,
                cost_args,
                dynamics_args,
                loss_args,
                self.trajax_ilqr_kwargs,
            )
            return (high_level_loss, high_level_grad)

        in_axes = (0, None) + self.loss_vmap
        vloss, vgrads = jax.vmap(func, in_axes=in_axes)(
            X, params, *batch_loss_args
        )
        avg_loss = jnp.mean(vloss)
        net_grads = jax.tree_map(lambda x: jnp.mean(x, axis=0), vgrads)
        return avg_loss, net_grads
