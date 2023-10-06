"""wasserstein distance based policy."""

import functools

import jax
import jax.numpy as jnp

from gan_mpc.gan import js_policy


class WMPC(js_policy.JS_MPC):
    @functools.partial(jax.jit, static_argnums=0)
    def critic_loss(self, true_xseq, pred_xseq, params, key):
        critic_params = params["critic_params"]

        true_score = self.critic_model.predict(true_xseq, critic_params)
        pred_score = self.critic_model.predict(pred_xseq, critic_params)
        w_loss = pred_score - true_score

        grad_penalty = self.get_grad_penalty(true_xseq, pred_xseq, params, key)
        return w_loss + self.grad_penalty_rate * grad_penalty

    @functools.partial(jax.jit, static_argnums=0)
    def generator_loss(self, xcseq, useq, params, true_xseq):
        del useq
        critic_params = params["critic_params"]
        x_size = true_xseq.shape[-1]
        xseq, _ = jnp.split(xcseq, [x_size], axis=-1)
        score = self.critic_model.predict(xseq, critic_params)
        return jnp.mean(-score)
