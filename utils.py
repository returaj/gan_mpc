"""other codes."""

import json
import os

import numpy as np
from dm_control import suite

from gan_mpc.config import load_config

_MAIN_DIR_PATH = os.path.dirname(__file__)


def get_dm_expert_env(name):
    domain, task = name.split("_")
    return suite.load(domain, task)


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
    return np.concatenate(flattern)


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

    X, Y = np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
    return X, Y


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
