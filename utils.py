"""other codes."""

import json
import os

import numpy as np
import orbax
from dm_control import suite
from flax.training import orbax_utils

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


def get_config(config_path=None):
    config_path = config_path or os.path.join(
        _MAIN_DIR_PATH, "config/hyperparameters.yaml"
    )
    return load_config.Config.from_yaml(config_path)


def get_expert_trajectories(config, num_trajectories=50, path=None):
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


def save_flax_trainstate(ckpt, dir_path, basename):
    check_or_create_dir(dir_path)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(
        os.path.join(dir_path, basename), ckpt, save_args=save_args
    )


def save_model(model, model_config, train_config, loss_dict, dir_path):
    abs_dir_path = os.path.join(_MAIN_DIR_PATH, dir_path)
    check_or_create_dir(abs_dir_path)
    dir_list = sorted(os.listdir(abs_dir_path), key=lambda x: -int(x))
    key = "0" if not dir_list else f"{int(dir_list[0]) + 1}"
    full_path = os.path.join(abs_dir_path, key)

    json_config = {
        "loss": loss_dict,
        "model": model_config.to_dict(),
        "train": train_config.to_dict(),
    }
    ckpt = {"model": model, "config": json_config}

    save_json(json_config, full_path, "config.json")
    save_flax_trainstate(ckpt, full_path, "model")
