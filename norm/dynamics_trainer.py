"""train dynamics model using env interactions."""

import collections
import functools

import jax
import jax.numpy as jnp
import optax

from gan_mpc import utils


def from_traj_to_seq(state_traj, action_traj, horizon, traj_len=1000):
    traj_len = min(traj_len, len(state_traj))
    num_elems = traj_len - horizon
    seq_states, seq_actions, seq_next_states = [], [], []
    for i in range(num_elems):
        seq_states.append(state_traj[i : i + horizon])
        seq_actions.append(action_traj[i : i + horizon])
        seq_next_states.append(state_traj[(i + 1) : (i + 1 + horizon)])
    return (
        jnp.array(seq_states),
        jnp.array(seq_actions),
        jnp.array(seq_next_states),
    )


def get_dataset(config, dataset_path):
    trajectories = utils.get_expert_trajectories(config, dataset_path)
    s_trajs, a_trajs = trajectories["states"], trajectories["actions"]
    horizon = config.mpc.horizon
    X, U, Y = [], [], []
    for s_traj, a_traj in zip(s_trajs, a_trajs):
        seq_states, seq_actions, seq_next_states = from_traj_to_seq(
            state_traj=s_traj,
            action_traj=a_traj,
            horizon=horizon,
            traj_len=config.mpc.train.dynamics.init_trajectory_len,
        )
        X.append(seq_states)
        U.append(seq_actions)
        Y.append(seq_next_states)
    return (
        jnp.concatenate(X, axis=0),
        jnp.concatenate(U, axis=0),
        jnp.concatenate(Y, axis=0),
    )


class ReplayBuffer:
    def __init__(self, horizon, maxlen):
        self.horizon = horizon
        self.states = collections.deque(maxlen=maxlen)
        self.actions = collections.deque(maxlen=maxlen)
        self.next_states = collections.deque(maxlen=maxlen)

    def add(self, state_traj, action_traj):
        seq_states, seq_actions, seq_next_states = from_traj_to_seq(
            state_traj, action_traj, self.horizon
        )
        self.states.extend(seq_states)
        self.actions.extend(seq_actions)
        self.next_states.extend(seq_next_states)

    def get_data(self):
        return (
            jnp.array(self.states),
            jnp.array(self.actions),
            jnp.array(self.next_states),
        )


@functools.partial(jax.jit, static_argnums=0)
def predict_loss(
    policy,
    params,
    xseq,
    useq,
    next_xseq,
    discount_factor,
    teacher_forcing,
):
    seqlen, xsize = xseq.shape

    @jax.jit
    def body(carry, t):
        xprev, dynamics_carry = carry
        x = jnp.where(teacher_forcing, xseq[t], xprev)
        xc = jnp.concatenate([x, dynamics_carry], axis=-1)
        next_xc = policy.dynamics(xc, useq[t], 0, params)
        next_x, dynamics_carry = jnp.split(next_xc, [xsize], axis=-1)
        return (next_x, dynamics_carry), next_x

    dynamics_carry = policy.get_carry(xseq[0])
    _, pred_next_xseq = jax.lax.scan(
        body, (xseq[0], dynamics_carry), jnp.arange(seqlen)
    )

    diff_square = (pred_next_xseq - next_xseq) ** 2
    return jnp.sum(utils.discounted_sum(diff_square, discount_factor))


@functools.partial(jax.jit, static_argnums=0)
def train_per_update(
    train_args,
    opt_state,
    params,
    perm,
    dataset,
    discount_factor,
    teacher_forcing,
):
    policy, opt = train_args
    X, U, Y = dataset

    partial_predict_loss = functools.partial(predict_loss, policy)

    @jax.jit
    def body(carry, p):
        params, opt_state = carry
        batch_x, batch_u, batch_y = X[p], U[p], Y[p]

        def loss_fn(params):
            losses = jax.vmap(
                partial_predict_loss, in_axes=(None, 0, 0, 0, None, None)
            )(
                params,
                batch_x,
                batch_u,
                batch_y,
                discount_factor,
                teacher_forcing,
            )
            return jnp.mean(losses)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, opt_state), losses = jax.lax.scan(body, (params, opt_state), perm)
    return params, opt_state, jnp.mean(losses)


def train_params(
    train_args,
    opt_state,
    params,
    dataset,
    num_updates,
    batch_size,
    discount_factor,
    teacher_forcing_factor,
    key,
    id,
):
    datasize = dataset[0].shape[0]
    steps_per_update = datasize // batch_size
    train_losses = []
    for up in range(1, num_updates + 1):
        key, subkey = jax.random.split(key)
        perm = jax.random.choice(
            subkey, datasize, shape=(steps_per_update, batch_size)
        )

        teacher_forcing = (id + up) <= (num_updates * teacher_forcing_factor)
        params, opt_state, train_loss = train_per_update(
            train_args=train_args,
            opt_state=opt_state,
            params=params,
            perm=perm,
            dataset=dataset,
            discount_factor=discount_factor,
            teacher_forcing=teacher_forcing,
        )
        train_losses.append(float(train_loss))
    return params, opt_state, train_losses


@utils.timeit
def train(
    env,
    train_args,
    opt_state,
    params,
    dataset,
    replay_buffer,
    num_episodes,
    max_interactions_per_episode,
    num_updates,
    batch_size,
    discount_factor,
    teacher_forcing_factor,
    key,
    id,
):
    policy, opt = train_args

    if id == 1:
        key, subkey = jax.random.split(key)
        params, opt_state, _ = train_params(
            train_args=(policy, opt),
            opt_state=opt_state,
            params=params,
            dataset=dataset,
            num_updates=3,
            batch_size=batch_size,
            discount_factor=discount_factor,
            teacher_forcing_factor=1.0,
            key=subkey,
            id=0,
        )

    episode_rewards = []
    episode_train_losses = []
    episode_test_losses = []
    for ep in range(1, num_episodes + 1):
        key, subkey = jax.random.split(key)
        state_traj, action_traj, _, rewards = utils.run_dm_policy(
            env=env,
            policy_fn=policy.get_optimal_action,
            params=params,
            max_interactions=max_interactions_per_episode,
        )
        replay_buffer.add(state_traj, action_traj)
        episode_rewards.append(rewards)

        replay_dataset = replay_buffer.get_data()
        params, opt_state, train_losses = train_params(
            train_args=(policy, opt),
            opt_state=opt_state,
            params=params,
            dataset=replay_dataset,
            num_updates=num_updates,
            batch_size=batch_size,
            discount_factor=discount_factor,
            teacher_forcing_factor=teacher_forcing_factor * num_episodes,
            key=subkey,
            id=(num_updates * (ep - 1)),
        )

        episode_train_losses.extend(train_losses)
    return (
        params,
        opt_state,
        replay_buffer,
        episode_rewards,
        episode_train_losses,
        episode_test_losses,
    )
