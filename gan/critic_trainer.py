"""critic parameters training."""

import functools

import jax
import jax.numpy as jnp
import optax

from gan_mpc import utils


@functools.partial(jax.jit, static_argnums=0)
def get_dataset(policy, params, true_dataset, key):
    @jax.jit
    def func(X, true_Y):
        datasize = true_Y.shape[0]
        true_label = jnp.ones(datasize, dtype=jnp.float32)
        xsize = X.shape[-1]

        def predict(x, params):
            xc, *_ = policy(x, params)
            y, _ = jnp.split(xc, [xsize], axis=-1)
            return y

        pred_Y = jax.vmap(predict, in_axes=(0, None))(X, params)
        pred_label = -1 * jnp.ones(datasize, dtype=jnp.float32)

        return (
            jnp.concatenate([true_Y, pred_Y], axis=0),
            jnp.concatenate([true_label, pred_label], axis=0),
        )

    true_train_data, true_test_data = true_dataset
    train_X, train_label = func(*true_train_data)
    test_X, test_label = func(*true_test_data)

    perm = jax.random.permutation(key, train_X.shape[0])
    return (train_X[perm], train_label[perm]), (test_X, test_label)


@functools.partial(jax.jit, static_argnums=0)
def calculate_loss(policy, params, dataset):
    X, Y = dataset
    losses = jax.vmap(policy.critic_loss, in_axes=(0, 0, None))(X, Y, params)
    return jnp.mean(losses)


@functools.partial(jax.jit, static_argnums=0)
def train_critic_parameters(train_args, opt_state, params, perm, dataset):
    policy, opt = train_args
    X, Y = dataset

    @jax.jit
    def body(carry, p):
        params, opt_state = carry
        batch_x, batch_y = X[p], Y[p]
        loss, grads = policy.critic_loss_and_grad(batch_x, batch_y, params)
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
    true_dataset,
    num_updates,
    batch_size,
    key,
    id,
):
    del id
    policy, opt = train_args
    key, subkey = jax.random.split(key)
    train_data, test_data = get_dataset(policy, params, true_dataset, subkey)
    datasize = train_data[0].shape[0]
    steps_per_update = datasize // batch_size
    train_losses, test_losses = [], []
    for _ in range(1, num_updates + 1):
        key, subkey = jax.random.split(key)
        perm = jax.random.choice(
            subkey, datasize, shape=(steps_per_update, batch_size)
        )
        params, opt_state, train_loss = train_critic_parameters(
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

    return params, opt_state, train_losses, test_losses
