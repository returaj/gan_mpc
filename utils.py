"""other codes."""

import numpy as np
from dm_control import suite


def get_dm_expert_env(name):
    domain, task = name.split("-")
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
