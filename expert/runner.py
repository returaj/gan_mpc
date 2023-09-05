"""runner code for expert prediction model."""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from gan_mpc import utils
from gan_mpc.expert import expert_model, trainer


def get_train_dataset(config, dataset_path=None, train_split=0.8):
    trajectories = utils.get_expert_trajectories(config, path=dataset_path)

    s_trajs, a_trajs = trajectories["states"], trajectories["actions"]
    seqlen = config.expert_prediction.train.seqlen
    states, actions, next_states = [], [], []
    for s_traj, a_traj in zip(s_trajs, a_trajs):
        traj_len, sdim = s_traj.shape
        _, adim = a_traj.shape
        num_elems = (traj_len - 1) // seqlen
        valid_size = num_elems * seqlen
        states.append(s_traj[:valid_size].reshape((num_elems, seqlen, sdim)))
        actions.append(a_traj[:valid_size].reshape((num_elems, seqlen, adim)))
        next_states.append(
            s_traj[1 : (valid_size + 1)].reshape((num_elems, seqlen, sdim))
        )
    states = jnp.concatenate(states, axis=0)
    actions = jnp.concatenate(actions, axis=0)
    next_states = jnp.concatenate(next_states, axis=0)

    data_size = states.shape[0]
    split_pos = int(train_split * data_size)
    key = jax.random.PRNGKey(config.seed)
    perm = jax.random.permutation(key, data_size)
    train_dataset = (
        states[perm[:split_pos]],
        actions[perm[:split_pos]],
        next_states[perm[:split_pos]],
    )
    test_dataset = (
        states[perm[split_pos:]],
        actions[perm[split_pos:]],
        next_states[perm[split_pos:]],
    )
    return train_dataset, test_dataset


def get_trainstate(model, params, tx):
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


def get_model(config, state_size, action_size):
    expert_model_config = config.expert_prediction.model
    model = expert_model.ExpertModel.get_model(
        model_config=expert_model_config, x_size=state_size, u_size=action_size
    )
    return model, expert_model_config


def get_params(config, model, state_size):
    seed = config.seed
    args = model.get_init_params(seed, 1, 1, state_size)
    return model.init(*args)


def get_optimizer(config):
    lr = config.expert_prediction.train.learning_rate
    return optax.chain(
        optax.clip_by_global_norm(max_norm=100.0), optax.adam(lr)
    )


def run(config_path=None):
    config = utils.get_config(config_path)
    env_type, env_name = config.env.type, config.env.expert.name
    state_size, action_size = utils.get_state_action_size(env_type, env_name)

    model, model_config = get_model(config, state_size, action_size)
    params = get_params(config, model, state_size)
    tx = get_optimizer(config)
    trainstate = get_trainstate(model, params, tx)

    dataset = get_train_dataset(config)

    train_config = config.expert_prediction.train
    trainstate, train_loss, test_loss = trainer.train(
        trainstate=trainstate,
        dataset=dataset,
        num_epochs=train_config.num_epochs,
        batch_size=train_config.batch_size,
        key=jax.random.PRNGKey(config.seed),
        discount_factor=train_config.discount_factor,
        teacher_forcing_factor=train_config.teacher_forcing_factor,
        print_step=train_config.print_step,
    )

    save_config = {
        "loss": {
            "train_loss": round(float(train_loss), 5),
            "test_loss": round(float(test_loss), 5),
        },
        "model": model_config.to_dict(),
        "train": train_config.to_dict(),
    }

    dir_path = f"trained_models/expert/{env_type}/{env_name}/"
    utils.save_all_args(dir_path, trainstate.params, save_config)


if __name__ == "__main__":
    run()
