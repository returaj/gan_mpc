"""Load dataset for cost_fn, dynamics_fn and expert_fn training and testing."""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from gan_mpc import utils


class DataLoader:
    def __init__(self, config, normalizer):
        self.config = config
        self.normalizer = normalizer
        self.expert_trajectories = None

    def get_expert_trajectories(self, path, num_trajectories, trajectory_len):
        with open(path, "r") as fp:
            data = json.load(fp)
        sample_data = {}
        trajs_reward = np.sum(data["rewards"], axis=1)
        # TODO(returaj) Please remove this magic number of 500,
        # this is done to ensure expert trajectories are proper.
        idx = np.argsort(-trajs_reward)
        idx = list(filter(lambda x: trajs_reward[x] > 500, idx))[
            :num_trajectories
        ]
        keys = ["states", "actions", "rewards"]
        for k, v in data.items():
            if k in keys:
                sample_data[k] = np.array(v)[idx, :trajectory_len]
        return sample_data

    def init(self):
        config = self.config
        env_type, env_name = config.env.type, config.env.expert.name
        trajectories_path = os.path.join(
            utils._MAIN_DIR_PATH,
            f"expert_trajectories/{env_type}/{env_name}/trajectories.json",
        )
        self.expert_trajectories = self.get_expert_trajectories(
            path=trajectories_path,
            num_trajectories=config.mpc.train.num_trajectories,
            trajectory_len=config.mpc.train.trajectory_len,
        )
        self.normalizer.update(
            state_dataset=self.expert_trajectories["states"],
            action_dataset=self.expert_trajectories["actions"],
        )

        # print reward mean and std of the expert trajectories
        rewards = np.sum(self.expert_trajectories["rewards"], axis=1)
        rwd_mean, rwd_std = np.mean(rewards), np.std(rewards)
        print(
            f"Expert trajectories reward mean: {rwd_mean:.3f} and reward std: {rwd_std:.3f}"
        )
        return self

    def shuffle_and_split_dataset(self, dataset, key, train_split=0.8):
        datasize = dataset[0].shape[0]
        split_pos = int(datasize * train_split)
        perm = jax.random.permutation(key, datasize)
        train_dataset, test_dataset = [], []
        for d in dataset:
            d = jnp.array(d)
            train_dataset.append(d[perm[:split_pos]])
            test_dataset.append(d[perm[split_pos:]])
        return tuple(train_dataset), tuple(test_dataset)

    def get_cost_dataset(self, key):
        if self.expert_trajectories is None:
            raise Exception(
                "Please call init before calling get_cost_dataset."
            )
        s_trajs = self.normalizer.normalize_state(
            self.expert_trajectories["states"]
        )
        horizon = self.config.mpc.horizon
        history = self.config.mpc.history
        X, Y = [], []
        for s_traj in s_trajs:
            traj_len, xsize = s_traj.shape
            num_elems = traj_len - horizon
            s_traj = np.concatenate(
                [np.zeros((history, xsize)), s_traj], axis=0
            )
            tmpX, tmpY = [], []
            for i in range(history, num_elems):
                tmpX.append(s_traj[i - history : i + 1])
                tmpY.append(s_traj[i : i + horizon + 1])
            X.append(tmpX)
            Y.append(tmpY)
        dataset = np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
        return self.shuffle_and_split_dataset(dataset, key)

    def get_dynamics_dataset(self, key):
        train_dataset, _ = self.get_expert_dataset(
            key, seqlen=self.config.mpc.horizon
        )
        return train_dataset

    def get_expert_dataset(self, key, seqlen=None):
        if self.expert_trajectories is None:
            raise Exception(
                "Please call init before calling get_cost_dataset."
            )
        s_trajs, a_trajs = self.normalizer.normalize(
            state_dataset=self.expert_trajectories["states"],
            action_dataset=self.expert_trajectories["actions"],
        )
        seqlen = seqlen or self.config.expert_prediction.train.seqlen
        X, U, Y = [], [], []
        for s_traj, a_traj in zip(s_trajs, a_trajs):
            traj_len = s_traj.shape[0]
            num_elems = traj_len - seqlen
            seq_states, seq_actions, seq_next_states = [], [], []
            for i in range(num_elems):
                seq_states.append(s_traj[i : i + seqlen])
                seq_actions.append(a_traj[i : i + seqlen])
                seq_next_states.append(s_traj[(i + 1) : (i + 1 + seqlen)])
            X.append(seq_states)
            U.append(seq_actions)
            Y.append(seq_next_states)
        dataset = (
            np.concatenate(X, axis=0),
            np.concatenate(U, axis=0),
            np.concatenate(Y, axis=0),
        )
        return self.shuffle_and_split_dataset(dataset, key)
