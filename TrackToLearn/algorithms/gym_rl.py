import torch

from typing import Tuple

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.environments.env import BaseEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GymRLAlgorithm(RLAlgorithm):

    def gym_train(
        self,
        env: BaseEnv,
    ) -> Tuple[float, float, float]:
        """
        Call the main training loop

        Parameters
        ----------
        env: BaseEnv
            The environment actions are applied on. Provides the state fed to
            the RL algorithm

        Returns
        -------
        actor_loss: float
            Cumulative policy training loss
        critic_loss: float
            Cumulative critic training loss
        running_reward: float
            Cummulative training steps reward
        """

        self.policy.train()

        state = env.reset()

        # Track forward
        _, reward, losses, length = \
            self._episode(state, env)

        return (
            losses,
            reward,
            length)

    def gym_validation(
        self,
        env: BaseEnv,
        render: bool = False,
    ) -> float:
        """
        Call the main loop

        Parameters
        ----------
        env: BaseEnv
            The environment actions are applied on. Provides the state fed to
            the RL algorithm

        Returns
        -------
        running_reward: float
            Cummulative training steps reward
        """
        # Switch policy to eval mode so no gradients are computed
        self.policy.eval()
        state = env.reset()
        if render:
            env.render()

        # Track forward
        _, reward = self._validation_episode(
            state, env)

        return reward
