import gym  # open ai gym
import numpy as np

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
        device=None,
        gamma=0.99,
        seed=1337,
        **kwargs,
    ):
        """
        Parameters
        ----------

        """
        self.n_envs = n_envs
        self._inner_envs = []
        for i in range(self.n_envs):
            env = gym.make(env_name, **kwargs)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(
                env, lambda reward: np.clip(reward, -10, 10))
            # env.seed(seed)
            # env.action_space.seed(seed)
            # env.observation_space.seed(seed)
            self._inner_envs.append(env)

        self.dones = np.asarray([False] * self.n_envs)

    def reset(self):
        states = np.asarray([
            self._inner_envs[i].reset()[0] for i in range(self.n_envs)])
        self.dones = np.asarray([False] * self.n_envs)
        return states

    def step(self, action):
        not_done = [not d for d in self.dones]
        indices = np.asarray(range(self.n_envs))
        indices = indices[not_done]
        ns, r, d, t, *_ = zip(*[
            self._inner_envs[j].step(
                action[i]) for i, j in enumerate(indices)])

        n_i, r_i, d_i, t_i = (
            np.asarray(ns), np.asarray(r), np.asarray(d), np.asarray(t))
        self.dones[indices[d_i]] = True
        self.dones[indices[t_i]] = True
        not_dones = [(not d) and (not t) for (d, t) in zip(d_i, t_i)]
        self.continue_idx = np.arange(len(d_i))[not_dones]
        return n_i, r_i, np.logical_or(d_i, t_i), {}

    def render(self, **kwargs):
        self._inner_envs[0].render(**kwargs)

    def harvest(self, states, compress=False):
        indices = np.asarray(range(self.n_envs))
        indices = indices[self.continue_idx]
        states = states[self.continue_idx]
        return states, indices

    def get_streamlines(self, compress=False):
        return None
