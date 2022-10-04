import d4rl
import gym
import torch

import numpy as np

from torch.utils.data import Dataset


class GymDataset(Dataset):
    """
    class that loads hdf5 dataset object
    """

    def __init__(
            self, env_name: str):
        """
        Args:
        """
        self.env_name = env_name
        self.data = self.get_dataset(env_name)

    def get_dataset(self, env_name):
        env = gym.make(env_name).unwrapped
        dataset = d4rl.qlearning_dataset(env)
        return dict(
            states=dataset['observations'],
            actions=dataset['actions'],
            next_states=dataset['next_observations'],
            rewards=dataset['rewards'],
            dones=dataset['terminals'].astype(np.float32),
        )
        return dataset

    def get_one_input(self):

        return self.data['states'][0]

    def __getitem__(self, index):
        """This method loads, transforms and returns slice corresponding to the
        corresponding index.
        :arg
            index: the index of the slice within patient data
        :return
            A tuple (input, target)
        """
        states = self.data['states'][index][None, ...]
        actions = self.data['actions'][index][None, ...]
        rewards = np.array([self.data['rewards'][index]])
        next_states = self.data['next_states'][index][None, ...]
        dones = np.array([self.data['dones'][index]], dtype=float)
        states, actions, rewards, next_states, dones = map(torch.from_numpy, [states, actions, rewards, next_states, dones])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        return the length of the dataset
        """
        return int(len(self.data['states']))
