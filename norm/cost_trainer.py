"""norm based cost parameters training."""

import jax
import jax.numpy as jnp
import optax

from gan_mpc import utils


def get_dataset(config, dataset_path, key, train_split=0.8):
    X, Y = utils.get_policy_training_dataset(config, dataset_path)
    X, Y = jnp.array(X), jnp.array(Y)
    data_size = X.shape[0]
    split_pos = int(data_size * train_split)
    _, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, data_size)
    train_dataset = (
        X[perm[:split_pos]],
        Y[perm[:split_pos]],
    )
    test_dataset = (
        X[perm[split_pos:]],
        Y[perm[split_pos:]],
    )
    return (train_dataset, test_dataset)


def calculate_loss(policy_args, dataset):
    policy, params = policy_args
    batch_x, batch_y = dataset

    func = jax.jit(lambda x: policy(x, params))
    pred_y, pred_u, *_ = jax.vmap(func, in_axes=(0,))(batch_x)
    batch_loss = jax.vmap(policy.loss, in_axes=(0, 0, 0))(
        pred_y, pred_u, batch_y
    )
    return jnp.mean(batch_loss)


def train_cost_parameters(
    policy_args,
    opt_args,
    perm,
    dataset,
):
    policy, params = policy_args
    opt, opt_state = opt_args
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
def train(policy_args, opt_args, dataset, num_updates, batch_size, key, id):
    del id
    policy, params = policy_args
    opt, opt_state = opt_args
    train_data, test_data = dataset
    datasize = train_data[0].shape[0]
    steps_per_update = datasize // batch_size
    train_losses, test_losses = [], []
    for _ in range(1, num_updates + 1):
        key, subkey = jax.random.split(key)
        perm = jax.random.choice(
            subkey, datasize, shape=(steps_per_update, batch_size)
        )
        params, opt_state, train_loss = train_cost_parameters(
            policy_args=(policy, params),
            opt_args=(opt, opt_state),
            perm=perm,
            dataset=train_data,
        )
        test_loss = calculate_loss(
            policy_args=(policy, params), dataset=test_data
        )
        train_losses.append(float(train_loss))
        test_losses.append(float(test_loss))

    return params, opt_state, train_losses, test_losses
