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
        entropy_rate=1e-3,
        grad_penalty_rate=10.0,
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
        self.entropy_rate = entropy_rate
        self.grad_penalty_rate = grad_penalty_rate
        self.critic_model = critic_model

    def init(
        self, mpc_weights, cost_args, dynamics_args, expert_args, critic_args
    ):
        params = super().init(
            mpc_weights, cost_args, dynamics_args, expert_args
        )
        params["critic_params"] = self.critic_model.init(*critic_args)
        return params

    @functools.partial(jax.jit, static_argnums=0)
    def get_grad_penalty(self, true_xseq, pred_xseq, params, key):
        critic_params = params["critic_params"]
        ep = jax.random.uniform(key)
        mix_xseq = ep * true_xseq + (1 - ep) * pred_xseq

        def fun(xseq):
            logit = self.critic_model.predict(xseq, critic_params)
            return jnp.sum(logit)  # reshape to scalar

        grad = jax.grad(fun)(mix_xseq)
        penalty = (jnp.linalg.norm(grad) - 1) ** 2
        return penalty

    @functools.partial(jax.jit, static_argnums=0)
    def critic_loss(self, true_xseq, pred_xseq, params, key):
        critic_params = params["critic_params"]

        true_logit = self.critic_model.predict(true_xseq, critic_params)
        pred_logit = self.critic_model.predict(pred_xseq, critic_params)
        true_loss = optax.sigmoid_binary_cross_entropy(
            true_logit, jnp.ones(true_logit.shape)
        )
        pred_loss = optax.sigmoid_binary_cross_entropy(
            pred_logit, jnp.zeros(pred_logit.shape)
        )
        ce_loss = (true_loss + pred_loss) / 2

        true_ent_loss = -get_logit_bernoulli_entropy(true_logit)
        pred_ent_loss = -get_logit_bernoulli_entropy(pred_logit)
        ent_loss = (true_ent_loss + pred_ent_loss) / 2

        grad_penalty = self.get_grad_penalty(true_xseq, pred_xseq, params, key)
        return (
            ce_loss
            + self.entropy_rate * ent_loss
            + self.grad_penalty_rate * grad_penalty
        )

    @functools.partial(jax.jit, static_argnums=0)
    def critic_loss_and_grad(
        self, batch_true_xseq, batch_pred_xseq, params, batch_key
    ):
        @jax.jit
        def loss_fn(params):
            losses = jax.vmap(self.critic_loss, in_axes=(0, 0, None, 0))(
                batch_true_xseq, batch_pred_xseq, params, batch_key
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
