"""policy for evaluation. Not to be used for training."""

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


class EvalMPC:
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
        def func(xc, useq, params, cost_args, dynamics_args):
            return opt.ilqr_solve(
                self.cost,
                self.dynamics,
                xc,
                useq,
                params,
                cost_args,
                dynamics_args,
                self.trajax_ilqr_kwargs,
            )

        return jax.jit(func)

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

    @functools.partial(jax.jit, static_argnums=0)
    def get_dynamics_carry(self, history_x, history_u, params):
        """
        history_x: (seqlen + 1, xsize)
        history_u: (seqlen, usize)
        """

        dynamics_params = params["dynamics_params"]
        return self.dynamics_model.get_history_carry(
            history_x[:-1], history_u, dynamics_params
        )

    @functools.partial(jax.jit, static_argnums=0)
    def get_goal_states_init_actions(self, histroy_x, params):
        """
        history_x: (seqlen + 1, xsize)
        """

        expert_params = params["expert_params"]
        x = histroy_x[-1]
        xseq = jnp.vstack(
            [x, jnp.zeros((self.config.mpc.horizon - 1, x.shape[0]))]
        )
        carry = self.expert_model.get_history_carry(
            histroy_x, xseq, expert_params
        )
        _, (
            goal_xseq,
            init_useq,
        ) = self.expert_model.get_carry_next_state_and_action_seq(
            carry, xseq, expert_params
        )
        return goal_xseq, init_useq

    @functools.partial(jax.jit, static_argnums=0)
    def get_optimal_values(self, params, history_x, history_u):
        """
        history_x: (history + 1, xsize)
        history_u: (history, usize)
        """

        goal_xseq, init_useq = self.get_goal_states_init_actions(
            history_x, params
        )
        init_carry = self.get_dynamics_carry(history_x, history_u, params)
        cost_args = (goal_xseq,)
        dynamics_args = ()
        x = history_x[-1]
        xc = jnp.concatenate([x, init_carry], axis=-1)
        return self.solver(xc, init_useq, params, cost_args, dynamics_args)

    def get_optimal_action(self, params, history_x, history_u):
        _, useq, *_ = self.get_optimal_values(params, history_x, history_u)
        return useq[0]
