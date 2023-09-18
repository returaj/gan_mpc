"""Base learnable MPC policy used only for test/training."""

import functools

import jax
import jax.numpy as jnp

from gan_mpc.policy import eval
from gan_mpc.policy import optimizers as opt


class BaseMPC(eval.EvalMPC):
    def __init__(
        self,
        config,
        cost_model,
        dynamics_model,
        expert_model,
        loss_vmap=(0,),
        trajax_ilqr_kwargs=eval.TRAJAX_iLQR_KWARGS,
    ):
        super().__init__(
            config=config,
            cost_model=cost_model,
            dynamics_model=dynamics_model,
            expert_model=expert_model,
            trajax_ilqr_kwargs=trajax_ilqr_kwargs,
        )
        self.loss_vmap = loss_vmap

    @functools.partial(jax.jit, static_argnums=0)
    def get_dynamics_carry(self, history_x, *args):
        """
        history_x: (seqlen + 1, xsize)
        """

        del args
        return self.dynamics_model.get_zero_carry(history_x[:-1])

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
        # change it to get_zero_carry if history is not present.
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
    def get_optimal_values(self, params, history_x, *args):
        """
        history_x: (seqlen + 1, xsize)
        """

        del args
        goal_xseq, init_useq = self.get_goal_states_init_actions(
            history_x, params
        )
        init_carry = self.get_dynamics_carry(history_x)
        cost_args = (goal_xseq,)
        dynamics_args = ()
        x = history_x[-1]
        xc = jnp.concatenate([x, init_carry], axis=-1)
        return self.solver(xc, init_useq, params, cost_args, dynamics_args)

    def get_optimal_action(self, params, history_x, *args):
        _, useq, *_ = self.get_optimal_values(params, history_x, *args)
        return useq[0]

    def loss(self, xcseq, useq, params, *args):
        raise NotImplementedError

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss_and_grad(self, history_X, params, batch_loss_args):
        """
        history_X: (batch, seqlen + 1, xsize)
        """

        @jax.jit
        def func(history_x, params, *loss_args):
            goal_xseq, init_useq = self.get_goal_states_init_actions(
                history_x, params
            )
            init_carry = self.get_dynamics_carry(history_x)
            cost_args = (goal_xseq,)
            dynamics_args = ()
            x = history_x[-1]
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
            history_X, params, *batch_loss_args
        )
        avg_loss = jnp.mean(vloss)
        net_grads = jax.tree_map(lambda x: jnp.mean(x, axis=0), vgrads)
        return avg_loss, net_grads
