"""other codes."""

import jax
import jax.numpy as jnp
import json
import os

import numpy as np
from dm_control import suite

from gan_mpc.config import load_config
from gan_mpc.cost import cost_model
from gan_mpc.cost import nn as cost_nn
from gan_mpc.dynamics import dynamics_model
from gan_mpc.dynamics import nn as dynamics_nn
from gan_mpc.expert import expert_model

_MAIN_DIR_PATH = os.path.dirname(__file__)


def get_dm_expert_env(name):
    domain, task = name.split("_")
    return suite.load(domain, task)


def get_dm_imitator_env(name, seed):
    domain, task = name.split("_")
    return suite.load(domain, task, task_kwargs={"random": seed})


def get_imitator_env(env_type, env_name, seed):
    if env_type == "brax":
        raise NotImplementedError
    elif env_type == "dmcontrol":
        return get_dm_imitator_env(env_name, seed)
    raise Exception(
        f"env_type can be either brax or dmcontrol, but given {env_type}"
    )


def get_dm_state_action_size(name):
    def get_size_from_spec(spec):
        size = 0
        for s in spec:
            size += np.int32(np.prod(s.shape))
        return size

    env = get_dm_expert_env(name)
    state_size = get_size_from_spec(env.observation_spec().values())
    action_size = get_size_from_spec([env.action_spec()])
    return state_size, action_size


def get_state_action_size(env_type, env_name):
    if env_type == "brax":
        raise NotImplementedError("brax environment has not been tested yet.")
    elif env_type == "dmcontrol":
        return get_dm_state_action_size(env_name)
    else:
        raise Exception(
            f"env_type can be either brax or dmcontrol, but given {env_type}"
        )


def flatten_tree_obs(obs):
    flattern = []
    for v in obs.values():
        v = np.array([v]) if np.isscalar(v) else np.ravel(v)
        flattern.append(v)
    return jnp.concatenate(flattern)


def get_config(config_path):
    config_path = os.path.join(_MAIN_DIR_PATH, config_path)
    return load_config.Config.from_yaml(config_path)


def get_expert_trajectories(config, path=None, num_trajectories=50):
    env_type, env_name = config.env.type, config.env.expert.name
    path = path or os.path.join(
        _MAIN_DIR_PATH,
        f"expert_trajectories/{env_type}/{env_name}/trajectories.json",
    )
    with open(path, "r") as fp:
        data = json.load(fp)
    sample_data = {}
    for k, v in data.items():
        sample_data[k] = np.array(v[:num_trajectories])
    return sample_data


def check_or_create_dir(path):
    if not os.path.exists(path=path):
        os.makedirs(path, exist_ok=True)


def save_json(data, dir_path, basename):
    dir_path = os.path.join(_MAIN_DIR_PATH, dir_path)
    check_or_create_dir(dir_path)
    with open(os.path.join(dir_path, basename), "w") as fp:
        json.dump(data, fp, indent=4, sort_keys=True)


def load_json(path):
    full_path = os.path.join(_MAIN_DIR_PATH, path)
    with open(full_path, "r") as fp:
        data = json.load(fp)
    return data


def save_params(params, config_dict, dir_path):
    abs_dir_path = os.path.join(_MAIN_DIR_PATH, dir_path)
    check_or_create_dir(abs_dir_path)
    dir_list = sorted(os.listdir(abs_dir_path), key=lambda x: -int(x))
    key = "0" if not dir_list else f"{int(dir_list[0]) + 1}"
    full_path = os.path.join(abs_dir_path, key)
    save_json(config_dict, full_path, "config.json")
    np.save(os.path.join(full_path, "params.npy"), params, allow_pickle=True)


def load_params(params_path, from_np=True):
    abs_params_path = os.path.join(_MAIN_DIR_PATH, params_path)
    if from_np:
        params = np.load(abs_params_path, allow_pickle=True).item()
    else:
        raise NotImplementedError("params must be saved using numpy.")
    return params


def get_masked_labels(all_vars, masked_vars, tx_key, zero_key):
    labels = {}
    for v in all_vars:
        if v in masked_vars:
            labels[v] = zero_key
        else:
            labels[v] = tx_key
    return labels


def get_policy_training_dataset(config, dataset_path=None):
    trajectories = get_expert_trajectories(
        config=config,
        path=dataset_path,
        num_trajectories=config.mpc.train.cost.num_trajectories,
    )

    s_trajs = trajectories["states"]
    horizon = config.mpc.horizon
    X, Y = [], []
    for s_traj in s_trajs:
        traj_len, _ = s_traj.shape
        num_elems = traj_len - horizon
        X.append(s_traj[:num_elems])
        tmp = []
        for i in range(num_elems):
            tmp.append(s_traj[i : i + horizon + 1])
        Y.append(tmp)

    return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)


def get_cost_model(config):
    model_config = config.mpc.model.cost
    mlp_config = model_config.mlp
    nn_model = cost_nn.MLP(
        num_layers=mlp_config.num_layers,
        num_hidden_units=mlp_config.num_hidden_units,
        fout=mlp_config.fout,
    )
    return cost_model.MujocoBasedModel(config, nn_model), model_config


def get_dynamics_model(config, x_size):
    model_config = config.mpc.model.dynamics
    mlp_config = model_config.mlp
    nn_model = dynamics_nn.MLP(
        num_layers=mlp_config.num_layers,
        num_hidden_units=mlp_config.num_hidden_units,
        x_out=x_size,
    )
    return dynamics_model.DynamicsModel(config, nn_model), model_config


def get_expert_model(config, x_size, u_size):
    env_type, env_name = config.env.type, config.env.expert.name
    env_id = config.mpc.model.expert.load_id
    saved_config_path = (
        f"trained_models/expert/{env_type}/{env_name}/{env_id}/config.json"
    )
    saved_config = load_json(saved_config_path)
    model_config = load_config.Config.from_dict(saved_config["model"])
    nn_model = expert_model.ExpertModel.get_model(
        model_config=model_config, x_size=x_size, u_size=u_size
    )
    return expert_model.ExpertModel(config, nn_model)


@jax.jit
def discounted_sum(mat, gamma):
    def body(t, val):
        curr_sum, discount = val
        curr_sum += discount * mat[t]
        discount *= gamma
        return (curr_sum, discount)

    length, m_size = mat.shape
    curr_sum, _ = jax.lax.fori_loop(0, length, body, (jnp.zeros(m_size), 1.0))
    return curr_sum


def run_dm_policy(env, policy, params, max_interactions):
    states, actions, next_states = [], [], []
    rewards = []
    jit_policy = jax.jit(lambda x: policy(x, params))
    timestep = env.reset()
    t = 0, 0
    while (not timestep.last()) and (t < max_interactions):
        x = flatten_tree_obs(timestep.observation)
        u = jit_policy(x)
        timestep = env.step(u)
        t += 1
        states.append(x)
        actions.append(u)
        rewards.append(timestep.reward)
    return (
        jnp.array(states),
        jnp.array(actions),
        rewards,
    )


"""Depricated

def save_flax_trainstate(ckpt, dir_path, basename):
    check_or_create_dir(dir_path)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(
        os.path.join(dir_path, basename), ckpt, save_args=save_args
    )

def save_model(model, config_dict, dir_path):
    abs_dir_path = os.path.join(_MAIN_DIR_PATH, dir_path)
    check_or_create_dir(abs_dir_path)
    dir_list = sorted(os.listdir(abs_dir_path), key=lambda x: -int(x))
    key = "0" if not dir_list else f"{int(dir_list[0]) + 1}"
    full_path = os.path.join(abs_dir_path, key)
    ckpt = {"model": model, "config": config_dict}
    save_json(config_dict, full_path, "config.json")
    save_flax_trainstate(ckpt, full_path, "model")
"""
