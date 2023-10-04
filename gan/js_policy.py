"""Jensson-Shannon based policy."""

import functools

import jax
import jax.numpy as jnp
import optax

from gan_mpc.policy import base, eval


def get_logit_bernoulli_entropy(logit):
    ent = (1 - jax.nn.sigmoid(logit)) * logit - jax.nn.log_sigmoid(logit)
    return ent


class JS_MPC(base.BaseMPC):
    def __init__(
        self,
        config,
        cost_model,
        dynamics_model,
        expert_model,
        critic_model,
        loss_vmap=(0,),
        entropy_coef=1e-3,
        trajax_ilqr_kwargs=eval.TRAJAX_iLQR_KWARGS,
    ):
        super().__init__(
            config,
            cost_model,
            dynamics_model,
            expert_model,
            loss_vmap,
            trajax_ilqr_kwargs,
        )
        self.entropy_coef = entropy_coef
        self.critic_model = critic_model

    def init(
        self, mpc_weights, cost_args, dynamics_args, expert_args, critic_args
    ):
        params = super().init(
            mpc_weights, cost_args, dynamics_args, expert_args
        )
        params["critic_params"] = self.critic_model.init(*critic_args)
        return params

    def critic_loss(self, xseq, label, params):
        critic_params = params["critic_params"]
        logit = self.critic_model.predict(xseq, critic_params)
        ce_loss = optax.sigmoid_binary_cross_entropy(logit, label)
        ent_loss = -get_logit_bernoulli_entropy(logit)
        return ce_loss + self.entropy_coef * ent_loss

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
        return jnp.mean(-jnp.log(p) + jnp.log(1 - p))

    def generator_loss_and_grad(self, batch_xseq, params, batch_loss_args):
        return self.loss_and_grad(batch_xseq, params, batch_loss_args)

    def loss(self, xcseq, useq, params, desired_xseq):
        return self.generator_loss(xcseq, useq, params, desired_xseq)
