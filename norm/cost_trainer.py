"""norm based cost parameters training."""

import functools

import jax
import jax.numpy as jnp
import optax

from gan_mpc import utils


@functools.partial(jax.jit, static_argnums=0)
def calculate_loss(policy, params, dataset):
    batch_x, batch_y = dataset

    func = jax.jit(lambda x: policy.get_optimal_values(params, x))
    pred_y, pred_u, *_ = jax.vmap(func, in_axes=(0,))(batch_x)
    batch_loss = jax.vmap(policy.loss, in_axes=(0, 0, None, 0))(
        pred_y, pred_u, params, batch_y
    )
    return jnp.mean(batch_loss)


@functools.partial(jax.jit, static_argnums=0)
def train_cost_parameters(
    train_args,
    opt_state,
    params,
    perm,
    dataset,
):
    policy, opt = train_args
    X, Y = dataset

    @jax.jit
    def body(carry, p):
        params, opt_state = carry
        batch_x, batch_y = X[p], Y[p]
        loss_args = (batch_y,)
        loss, grads = policy.loss_and_grad(batch_x, params, loss_args)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, opt_state), batch_loss = jax.lax.scan(
        body, (params, opt_state), perm
    )
    return params, opt_state, jnp.mean(batch_loss)


@utils.timeit
def train(
    train_args,
    opt_state,
    params,
    dataset,
    num_updates,
    batch_size,
    polyak_factor,
    key,
    id,
):
    del id
    policy, opt = train_args
    train_data, test_data = dataset
    prev_params = params
    datasize = train_data[0].shape[0]
    steps_per_update = datasize // batch_size
    train_losses, test_losses = [], []
    for _ in range(1, num_updates + 1):
        key, subkey = jax.random.split(key)
        perm = jax.random.choice(
            subkey, datasize, shape=(steps_per_update, batch_size)
        )
        params, opt_state, train_loss = train_cost_parameters(
            train_args=(policy, opt),
            opt_state=opt_state,
            params=params,
            perm=perm,
            dataset=train_data,
        )
        test_loss = calculate_loss(
            policy=policy, params=params, dataset=test_data
        )
        train_losses.append(float(train_loss))
        test_losses.append(float(test_loss))

    params = jax.tree_map(
        lambda x, y: polyak_factor * x + (1 - polyak_factor) * y,
        prev_params,
        params,
    )
    return params, opt_state, train_losses, test_losses
