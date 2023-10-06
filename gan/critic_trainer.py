"""critic parameters training."""

import functools

import jax
import jax.numpy as jnp
import optax

from gan_mpc import utils


def split_dataset(dataset, key):
    X, Y = dataset
    datasize = Y.shape[0]
    halfsize = datasize // 2
    perm = jax.random.choice(key, datasize, shape=(2, halfsize))
    true_data = X[perm[0]], Y[perm[0]]
    pred_data = X[perm[1]], Y[perm[1]]
    return (true_data, pred_data)


@functools.partial(jax.jit, static_argnums=0)
def get_pred_dataset(policy, params, pred_X):
    xsize = pred_X.shape[-1]

    def predict(x):
        xc, *_ = policy.get_optimal_values(params, x)
        y, _ = jnp.split(xc, [xsize], axis=-1)
        return y

    pred_Y = jax.vmap(predict)(pred_X)
    return pred_Y


@functools.partial(jax.jit, static_argnums=0)
def calculate_loss(policy, params, dataset):
    (_, true_y), (pred_x, _) = dataset
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, true_y.shape[0])
    pred_y = get_pred_dataset(policy, params, pred_x)
    losses = jax.vmap(policy.critic_loss, in_axes=(0, 0, None, 0))(
        true_y, pred_y, params, keys
    )
    return jnp.mean(losses)


@functools.partial(jax.jit, static_argnums=0)
def train_critic_parameters(train_args, opt_state, params, perm, key, dataset):
    policy, opt = train_args
    (_, true_Y), (pred_X, _) = dataset

    batch_size = perm.shape[1]

    @jax.jit
    def body(carry, p):
        params, opt_state, key = carry
        key, subkey = jax.random.split(key)
        batch_true_y, batch_pred_x = true_Y[p], pred_X[p]
        batch_pred_y = get_pred_dataset(policy, params, batch_pred_x)
        batch_key = jax.random.split(subkey, batch_size)
        loss, grads = policy.critic_loss_and_grad(
            batch_true_y, batch_pred_y, params, batch_key
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, key), loss

    (params, opt_state, _), batch_loss = jax.lax.scan(
        body, (params, opt_state, key), perm
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
    if id < 5:
        num_updates = 20  # warmup

    policy, opt = train_args
    key, subkey = jax.random.split(key)
    true_train_data, true_test_data = true_dataset
    train_data = split_dataset(true_train_data, subkey)
    test_data = split_dataset(true_test_data, subkey)
    datasize = train_data[0][0].shape[0]
    steps_per_update = datasize // batch_size
    train_losses, test_losses = [], []
    for _ in range(1, num_updates + 1):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        perm = jax.random.choice(
            subkey1, datasize, shape=(steps_per_update, batch_size)
        )
        params, opt_state, train_loss = train_critic_parameters(
            train_args=(policy, opt),
            opt_state=opt_state,
            params=params,
            perm=perm,
            dataset=train_data,
            key=subkey2,
        )
        test_loss = calculate_loss(
            policy=policy, params=params, dataset=test_data
        )
        train_losses.append(float(train_loss))
        test_losses.append(float(test_loss))

    return params, opt_state, train_losses, test_losses
