"""Jensson-Shannon based policy."""

import functools

import jax
import jax.numpy as jnp

from gan_mpc.policy import base


class JS_MPC(base.BaseMPC):
    def __init__(
        self,
        config,
        cost_model,
        dynamics_model,
        expert_model,
        critic_model,
        loss_vmap=(0,),
        trajax_ilqr_kwargs=base.TRAJAX_iLQR_KWARGS,
    ):
        super().__init__(
            config,
            cost_model,
            dynamics_model,
            expert_model,
            loss_vmap,
            trajax_ilqr_kwargs,
        )
        self.critic_model = critic_model

    def init(
        self, mpc_weights, cost_args, dynamics_args, expert_args, critic_args
    ):
        params = super().init(
            mpc_weights, cost_args, dynamics_args, expert_args
        )
        params["critic_params"] = self.critic_model.init(*critic_args)

    def critic_loss(self, xseq, label, params):
        critic_params = params["critic_params"]
        score = self.critic_model.predict(xseq, critic_params)
        p = jax.nn.sigmoid(score)
        p = jnp.where(label > 0, p, 1 - p)
        return -jnp.log(p)

    @functools.partial(jax.jit, static_argnums=0)
    def critic_loss_and_grad(self, batch_xseq, batch_label, params):
        @jax.jit
        def loss_fn(params):
            losses = jax.vmap(self.critic_loss, in_axes=(0, 0, None))(
                batch_xseq, batch_label, params
            )
            return jnp.mean(losses)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        return loss, grads

    @functools.partial(jax.jit, static_argnums=0)
    def generator_loss(self, xcseq, useq, params, actual_xseq):
        del useq
        critic_params = params["critic_params"]
        x_size = actual_xseq.shape[-1]
        xseq, _ = jnp.split(xcseq, [x_size], axis=-1)
        score = self.critic_model.predict(xseq, critic_params)
        p = jax.nn.sigmoid(score)
        return -jnp.log(p) + jnp.log(1 - p)

    def generator_loss_and_grad(self, batch_xseq, params, batch_loss_args):
        return self.loss_and_grad(batch_xseq, params, batch_loss_args)

    def loss(self, XC, U, params, desired_X):
        return self.generator_loss(XC, U, params, desired_X)
