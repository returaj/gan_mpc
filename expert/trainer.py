"""training code for expert model"""

import jax
import jax.numpy as jnp


@jax.jit
def discounted_sum(mat, gamma):
    def body(t, val):
        curr_sum, discount = val
        curr_sum += discount * mat[t]
        discount *= gamma
        return (curr_sum, discount)

    length = mat.shape[0]
    curr_sum, _ = jax.lax.fori_loop(0, length, body, (0, 1.0))
    return curr_sum


@jax.jit
def train_epoch(trainstate, perm, dataset, discount_factor):
    s, a, next_s = dataset
    steps_per_epoch = perm.shape[0]

    batch_discount_sum_fn = jax.vmap(discounted_sum, in_axes=(0, None))

    @jax.jit
    def body(trainstate, t):
        p = perm[t]
        batch_s, batch_a, batch_next_s = s[p], a[p], next_s[p]

        def loss_fn(params):
            pred_a, pred_next_s = trainstate.apply_fn(params, batch_s)
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
                    batch_discount_sum_fn(
                        next_s_diff_squared, discount_factor
                    ),
                    axis=1,
                )
            )
            return u_loss + next_s_loss

        loss, grads = jax.value_and_grad(loss_fn)(trainstate.params)
        trainstate = trainstate.apply_gradients(grads=grads)
        return trainstate, loss

    trainstate, batch_loss = jax.lax.scan(
        body, trainstate, jnp.arange(steps_per_epoch)
    )
    return trainstate, jnp.mean(batch_loss)


def train(
    trainstate,
    dataset,
    num_epochs,
    batch_size,
    key,
    discount_factor,
    print_step=10,
):
    train_data, _ = dataset
    datasize = train_data[0].shape[0]
    steps_per_epoch = datasize // batch_size
    epoch_loss = []
    for ep in range(1, num_epochs):
        key, subkey = jax.random.split(key)
        perm = jnp.random.choice(
            subkey, datasize, shape=(steps_per_epoch * batch_size,)
        )
        perm = perm.reshape((steps_per_epoch, batch_size))
        train_state, loss = train_epoch(
            trainstate, perm, train_data, discount_factor
        )
        if (ep % print_step) == 0:
            print(f"epoch: {ep} training_loss: {loss:.3f}")
        epoch_loss.append(loss)
    return trainstate, epoch_loss
