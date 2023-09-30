"""state normalizer class."""

import numpy as np


class BaseNormalizer:
    def update(self, dataset, *args, **kwargs):
        raise NotImplementedError

    def normalize(self, dataset, *args, **kwargs):
        raise NotImplementedError


class IdentityNormalizer(BaseNormalizer):
    def update(self, dataset, *args, **kwargs):
        del dataset, args, kwargs
        pass

    def normalize(self, dataset):
        return np.array(dataset)


class StandardNormalizer(BaseNormalizer):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def update(self, dataset):
        """
        updates mean and standard deviation

        Args:
            dataset (nd.array): size: (.., x_size)
        """
        dataset = np.array(dataset)
        axis = tuple(i for i in range(len(dataset.shape) - 1))
        self.mean = np.mean(dataset, axis=axis)
        self.std = np.std(dataset, axis=axis)

    def normalize(self, dataset):
        dataset = np.array(dataset)
        return (dataset - self.mean) / self.std


class JointNormalizer(BaseNormalizer):
    def __init__(
        self,
        state_normalizer: BaseNormalizer,
        action_normalizer: BaseNormalizer,
    ):
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer

    def update(self, state_dataset, action_dataset):
        self.state_normalizer.update(state_dataset)
        self.action_normalizer.update(action_dataset)

    def normalize_state(self, state_dataset):
        return self.state_normalizer.normalize(state_dataset)

    def normalize_action(self, action_dataset):
        return self.action_normalizer.normalize(action_dataset)

    def normalize(self, state_dataset, action_dataset):
        return (
            self.normalize_state(state_dataset),
            self.normalize_action(action_dataset),
        )
