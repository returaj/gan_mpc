"""training code for expert model"""

import jax
import jax.numpy as jnp

from gan_mpc import utils


@jax.jit
def calculate_loss(
    trainstate, params, dataset, discount_factor, teacher_forcing
):
    batch_discount_sum_fn = jax.vmap(utils.discounted_sum, in_axes=(0, None))

    batch_s, batch_a, batch_next_s = dataset
    pred_next_s, pred_a = trainstate.apply_fn(params, batch_s, teacher_forcing)
    u_diff_squared = (batch_a - pred_a) ** 2
    u_loss = jnp.mean(
        jnp.sum(
            batch_discount_sum_fn(u_diff_squared, discount_factor),
            axis=1,
        )
    )
    next_s_diff_squared = (batch_next_s - pred_next_s) ** 2
    next_s_loss = jnp.mean(
        jnp.sum(
            batch_discount_sum_fn(next_s_diff_squared, discount_factor),
            axis=1,
        )
    )
    return u_loss + next_s_loss


@utils.time
@jax.jit
def train_epoch(trainstate, perm, dataset, discount_factor, teacher_forcing):
    s, a, next_s = dataset

    @jax.jit
    def body(trainstate, p):
        batch_dataset = s[p], a[p], next_s[p]

        def loss_fn(params):
            return calculate_loss(
                trainstate,
                params,
                batch_dataset,
                discount_factor,
                teacher_forcing,
            )

        loss, grads = jax.value_and_grad(loss_fn)(trainstate.params)
        trainstate = trainstate.apply_gradients(grads=grads)
        return trainstate, loss

    trainstate, batch_loss = jax.lax.scan(body, trainstate, perm)
    return trainstate, jnp.mean(batch_loss)


def train(
    trainstate,
    dataset,
    num_epochs,
    batch_size,
    key,
    discount_factor,
    teacher_forcing_factor,
    print_step=10,
):
    train_data, test_data = dataset
    datasize = train_data[0].shape[0]
    steps_per_epoch = datasize // batch_size
    epoch_loss = []
    for ep in range(1, num_epochs + 1):
        key, subkey = jax.random.split(key)
        perm = jax.random.choice(
            subkey, datasize, shape=(steps_per_epoch, batch_size)
        )
        teacher_forcing = ep <= (num_epochs * teacher_forcing_factor)
        trainstate, train_loss, exe_time = train_epoch(
            trainstate, perm, train_data, discount_factor, teacher_forcing
        )
        if (ep % print_step) == 0:
            test_loss = calculate_loss(
                trainstate,
                trainstate.params,
                test_data,
                discount_factor,
                teacher_forcing=False,
            )
            print(
                f"epoch: {ep} exe_time: {exe_time:.2f} mins, "
                f"training_loss: {train_loss:.4f} test_loss: {test_loss:.4f}"
            )
        epoch_loss.append(train_loss)

    test_loss = calculate_loss(
        trainstate,
        trainstate.params,
        test_data,
        discount_factor,
        teacher_forcing=False,
    )
    return (
        trainstate,
        epoch_loss[-1],
        test_loss,
    )
