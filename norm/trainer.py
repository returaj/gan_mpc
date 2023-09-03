"""norm based policy training."""

import jax
import jax.numpy as jnp
import optax


def calculate_loss(policy_args, dataset):
    policy, params = policy_args
    batch_x, batch_y = dataset

    func = jax.jit(lambda x: policy(x, params))
    pred_y, pred_u, *_ = jax.vmap(func, in_axes=(0,))(batch_x)
    batch_loss = jax.vmap(policy.loss, in_axes=(0, 0, 0))(
        pred_y, pred_u, batch_y
    )
    return jnp.mean(batch_loss)


def train_policy_cost(
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
        loss_vmap, loss_args = (0,), (batch_y,)
        loss, grads = policy.loss_and_grad(
            batch_x, params, loss_vmap, loss_args
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, opt_state), batch_loss = jax.lax.scan(
        body, (params, opt_state), perm
    )
    return params, opt_state, jnp.mean(batch_loss)


def train(
    policy_args,
    opt_args,
    dataset,
    num_epochs,
    batch_size,
    key,
    print_step=10,
):
    policy, params = policy_args
    opt, opt_state = opt_args
    train_data, test_data = dataset
    datasize = train_data[0].shape[0]
    steps_per_epoch = datasize // batch_size
    epoch_loss = []
    for ep in range(1, num_epochs + 1):
        key, subkey = jax.random.split(key)
        perm = jax.random.choice(
            subkey, datasize, shape=(steps_per_epoch, batch_size)
        )
        params, opt_state, train_loss = train_policy_cost(
            policy_args=(policy, params),
            opt_args=(opt, opt_state),
            perm=perm,
            dataset=train_data,
        )
        if (ep % print_step) == 0:
            test_loss = calculate_loss(
                policy_args=(policy, params), dataset=test_data
            )
            print(
                f"epoch: {ep} training_loss: {train_loss:.4f} test_loss: {test_loss:.4f}"
            )
        epoch_loss.append(train_loss)

    test_loss = calculate_loss(policy_args=(policy, params), dataset=test_data)
    return params, epoch_loss[-1], test_loss
