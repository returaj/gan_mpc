"""other codes."""

import json
import os
import re
import time

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from dm_control import suite

from gan_mpc.config import load_config
from gan_mpc.cost import cost_model
from gan_mpc.cost import nn as cost_nn
from gan_mpc.critic import critic_model
from gan_mpc.critic import nn as critic_nn
from gan_mpc.dynamics import dynamics_model
from gan_mpc.dynamics import nn as dynamics_nn
from gan_mpc.expert import expert_model

_MAIN_DIR_PATH = os.path.dirname(__file__)


def timeit(fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        ret = fn(*args, **kwargs)
        exe_time = (time.time() - start_time) / 60
        if isinstance(ret, tuple):
            return *ret, exe_time
        return ret, exe_time

    return wrapper_fn


def get_dm_expert_env(name):
    domain, task = name.split("_")
    return suite.load(domain, task)


def get_dm_imitator_env(seed, name, env_physics):
    domain, task = name.split("_")
    env = suite.load(domain, task, task_kwargs={"random": seed})
    body_mass_pattern = re.compile("^body_mass_(.*)$")
    geom_size_pattern = re.compile("^geom_size_(.*)$")
    jnt_stiffness_pattern = re.compile("^jnt_stiffness_(.*)$")
    for kv in env_physics:
        key, value = kv["key"], int(kv["value"])
        mass_key = body_mass_pattern.search(key)
        if mass_key:
            env.physics.named.model.body_mass[mass_key.group(1)] *= value
        geom_key = geom_size_pattern.search(key)
        if geom_key:
            env.physics.named.model.geom_size[geom_key.group(1)] *= value
        stiffness_key = jnt_stiffness_pattern.search(key)
        if stiffness_key:
            env.physics.named.model.jnt_stiffness[
                stiffness_key.group(1)
            ] *= value
    return env


def get_imitator_env(config):
    seed = config.seed
    env_type, env_name = config.env.type, config.env.imitator.name
    env_physics = config.env.imitator.physics
    if env_type == "brax":
        raise NotImplementedError
    elif env_type == "dmcontrol":
        return get_dm_imitator_env(seed, env_name, env_physics)
    raise Exception(
        f"env_type can be either brax or dmcontrol, but given {env_type}"
    )


def get_size_from_dm_spec(spec):
    size = 0
    for s in spec:
        size += np.int32(np.prod(s.shape))
    return size


def get_dm_state_action_size(name):
    env = get_dm_expert_env(name)
    state_size = get_size_from_dm_spec(env.observation_spec().values())
    action_size = get_size_from_dm_spec([env.action_spec()])
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


def save_all_args(dir_path, params, model_config, *other_json_args):
    abs_dir_path = os.path.join(_MAIN_DIR_PATH, dir_path)
    check_or_create_dir(abs_dir_path)
    dir_list = sorted(os.listdir(abs_dir_path), key=lambda x: -int(x))
    key = "0" if not dir_list else f"{int(dir_list[0]) + 1}"
    full_path = os.path.join(abs_dir_path, key)
    # save model params
    save_json(model_config, full_path, "config.json")
    np.save(os.path.join(full_path, "params.npy"), params, allow_pickle=True)
    # save json data
    for json_data, name in other_json_args:
        save_json(json_data, full_path, name)
    return full_path


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
    if model_config.use == "lstm":
        lstm_config = model_config.lstm
        nn_model = dynamics_nn.LSTM(
            lstm_features=lstm_config.lstm_features,
            num_layers=lstm_config.num_layers,
            num_hidden_units=lstm_config.num_hidden_units,
            x_out=x_size,
        )
    elif model_config.use == "mlp":
        mlp_config = model_config.mlp
        nn_model = dynamics_nn.MLP(
            num_layers=mlp_config.num_layers,
            num_hidden_units=mlp_config.num_hidden_units,
            x_out=x_size,
        )
    else:
        raise ValueError("Choose either mlp or lstm model.")
    return dynamics_model.DynamicsModel(config, nn_model), model_config


def get_critic_model(config):
    model_config = config.mpc.model.critic
    if model_config.use == "lstm":
        lstm_config = model_config.lstm
        nn_model = critic_nn.LSTM(
            lstm_features=lstm_config.lstm_features,
            num_layers=lstm_config.num_layers,
            num_hidden_units=lstm_config.num_hidden_units,
        )
    else:
        raise ValueError("Choose lstm model.")
    return critic_model.CriticModel(config, nn_model), model_config


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


def save_video(env, policy_fn, params, buffer, dir_path, file_path):
    abs_path = os.path.join(_MAIN_DIR_PATH, dir_path, file_path)
    _, _, frames, _ = run_dm_policy(
        env, policy_fn, params, buffer, 1000, with_frames=True
    )
    writer = imageio.get_writer(abs_path, fps=30)
    for f in frames:
        writer.append_data(f)
    writer.close()


def run_dm_policy(
    env,
    policy_fn,
    params,
    buffer,
    max_interactions,
    with_frames=False,
):
    states, actions, rewards = [], [], []
    frames = []
    state_size = get_size_from_dm_spec(env.observation_spec().values())
    action_size = get_size_from_dm_spec([env.action_spec()])
    buffer.clear()
    buffer.append_state(jnp.zeros(state_size))
    buffer.append_action(jnp.zeros(action_size))
    timestep = env.reset()
    t = 0
    while (not timestep.last()) and (t < max_interactions):
        x = flatten_tree_obs(timestep.observation)
        buffer.append_state(x)
        u = policy_fn(
            params, buffer.get_state_data(), buffer.get_action_data()
        )
        buffer.append_action(u)
        timestep = env.step(u)
        t += 1
        if with_frames and (len(frames) < env.physics.data.time * 30):
            frames.append(env.physics.render(camera_id=0, width=240))
        states.append(x)
        actions.append(u)
        rewards.append(timestep.reward)
    return (
        np.array(states),
        np.array(actions),
        frames,
        rewards,
    )


def avg_run_dm_policy(
    env, policy_fn, params, buffer, num_runs, max_interactions
):
    avg_reward = 0.0
    for run in range(1, num_runs + 1):
        _, _, _, rwd_list = run_dm_policy(
            env=env,
            policy_fn=policy_fn,
            params=params,
            buffer=buffer,
            max_interactions=max_interactions,
        )
        avg_reward += (sum(rwd_list) - avg_reward) / run
    return avg_reward


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

def save_params(params, config_dict, dir_path):
    abs_dir_path = os.path.join(_MAIN_DIR_PATH, dir_path)
    check_or_create_dir(abs_dir_path)
    dir_list = sorted(os.listdir(abs_dir_path), key=lambda x: -int(x))
    key = "0" if not dir_list else f"{int(dir_list[0]) + 1}"
    full_path = os.path.join(abs_dir_path, key)
    save_json(config_dict, full_path, "config.json")
    np.save(os.path.join(full_path, "params.npy"), params, allow_pickle=True)
"""
