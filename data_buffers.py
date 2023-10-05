"""Different types of buffers."""

import collections

import numpy as np


class Buffer:
    def __init__(self, maxlen, normalizer):
        self.x_queue = collections.deque(maxlen=maxlen + 1)
        self.u_queue = collections.deque(maxlen=maxlen)
        self.normalizer = normalizer

    def append_state(self, x, *args):
        x = self.normalizer.normalize_state(x)
        self.x_queue.append(x)

    def append_action(self, u, *args):
        u = self.normalizer.normalize_action(u)
        self.u_queue.append(u)

    def get_state_data(self):
        return np.array(self.x_queue)

    def get_action_data(self):
        return np.array(self.u_queue)

    def clear(self):
        self.x_queue.clear()
        self.u_queue.clear()


class ReplayBuffer:
    def __init__(self, horizon, q_maxlen, normalizer):
        self.horizon = horizon
        self.state_queue = collections.deque(maxlen=q_maxlen)
        self.action_queue = collections.deque(maxlen=q_maxlen)
        self.next_state_queue = collections.deque(maxlen=q_maxlen)
        self.normalizer = normalizer

    def clear(self):
        self.state_queue.clear()
        self.action_queue.clear()
        self.next_state_queue.clear()

    def from_traj_to_seq(self, state_traj, action_traj):
        traj_len = len(state_traj)
        num_elems = traj_len - self.horizon
        seq_states, seq_actions, seq_next_states = [], [], []
        for i in range(num_elems):
            seq_states.append(state_traj[i : i + self.horizon])
            seq_actions.append(action_traj[i : i + self.horizon])
            seq_next_states.append(
                state_traj[(i + 1) : (i + 1 + self.horizon)]
            )
        return (
            np.array(seq_states),
            np.array(seq_actions),
            np.array(seq_next_states),
        )

    def add(self, state_traj, action_traj):
        state_traj = self.normalizer.normalize_state(state_traj)
        action_traj = self.normalizer.normalize_action(action_traj)
        seq_states, seq_actions, seq_next_states = self.from_traj_to_seq(
            state_traj, action_traj
        )
        self.state_queue.extend(seq_states)
        self.action_queue.extend(seq_actions)
        self.next_state_queue.extend(seq_next_states)

    def get_dataset(self):
        return (
            np.array(self.state_queue),
            np.array(self.action_queue),
            np.array(self.next_state_queue),
        )
