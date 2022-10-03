import gym  # open ai gym
import numpy as np
import pybulletgym  # noqa F401

from TrackToLearn.environments.env import BaseEnv


class GymWrapper(BaseEnv):
    """
    Abstract tracking environment.
    TODO: Add more explanations
    """

    def __init__(
        self,
        env_name: str,
        n_envs: int,
        device=None
    ):
        """
        Parameters
        ----------

        """
        self.n_envs = n_envs
        self._inner_envs = [gym.make(env_name) for i in range(self.n_envs)]
        self.dones = np.asarray([False] * self.n_envs)

    def reset(self):
        states = np.asarray([
            self._inner_envs[i].reset() for i in range(self.n_envs)])
        self.dones = np.asarray([False] * self.n_envs)
        return states

    def step(self, action):
        not_done = [not d for d in self.dones]
        indices = np.asarray(range(self.n_envs))
        indices = indices[not_done]
        ns, r, d, info = zip(*[
            self._inner_envs[j].step(action[i]) for i, j in enumerate(indices)])
        n_i, r_i, d_i = np.asarray(ns), np.asarray(r), np.asarray(d)
        self.dones[indices[d_i]] = True
        not_dones = [not d for d in d_i]
        self.continue_idx = np.arange(len(d_i))[not_dones]
        return n_i, r_i, d_i, {}

    def render(self, **kwargs):
        self._inner_envs[0].render(**kwargs)

    def get_streamlines(self, **kwargs):
        return []

    def close(self, **kwargs):
        self._inner_envs[0].close(**kwargs)

    def harvest(self, states, compress=False):
        indices = np.asarray(range(self.n_envs))
        indices = indices[self.continue_idx]
        tractogram = []
        states = states[self.continue_idx]
        return states, tractogram, indices
