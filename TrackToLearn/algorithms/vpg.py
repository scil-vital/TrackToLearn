import numpy as np
import torch

from collections import defaultdict

from typing import Tuple

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.algorithms.shared.onpolicy import PolicyGradient
from TrackToLearn.algorithms.shared.replay import ReplayBuffer
from TrackToLearn.algorithms.shared.utils import (
    add_item_to_means, mean_losses)
from TrackToLearn.environments.env import BaseEnv


class VPG(RLAlgorithm):
    """
    The sample-gathering and training algorithm.
    Based on:

        Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999).
        Policy gradient methods for reinforcement learning with function
        approximation. Advances in neural information processing systems, 12.

    Ratio and clipping were removed from PPO to obtain VPG.

    Some alterations have been made to the algorithms so it could be fitted to
    the tractography problem.

    """

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_dims: str,
        action_std: float = 0.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_loss_coeff: float = 0.0001,
        max_traj_length: int = 1,
        n_actors: int = 4096,
        rng: np.random.RandomState = None,
        device: torch.device = "cuda:0",
    ):
        """
        Parameters
        ----------
        input_size: int
            Input size for the model
        action_size: int
            Output size for the actor
        hidden_dims: str
            Widths and layers of the NNs
        lr: float
            Learning rate for optimizer
        gamma: float
            Gamma parameter future reward discounting
        entropy_loss_coeff: float
            Entropy bonus for the actor loss
        max_traj_length: int
            Maximum trajectory length to store in memory.
        n_actors: int
            Number of learners
        rng: np.random.RandomState
            rng for randomness. Should be fixed with a seed
        device: torch.device,
            Device to use for processing (CPU or GPU)
        """

        self.input_size = input_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.max_traj_length = max_traj_length
        self.entropy_loss_coeff = entropy_loss_coeff

        # Declare policy
        self.policy = PolicyGradient(
            input_size, action_size, hidden_dims, device, action_std
        ).to(device)

        # Optimizer for actor
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            input_size, action_size, n_actors, max_traj_length,
            self.gamma, lmbda=0.)

        self.on_policy = True
        self.max_action = 1.
        self.t = 1

        self.n_actors = n_actors
        self.rng = rng
        self.device = device

    def _episode(
        self,
        initial_state: np.ndarray,
        env: BaseEnv,
    ) -> Tuple[float, float, float, int]:
        """
        Main loop for the algorithm
        From a starting state, run the model until the env. says its done
        Gather transitions and train on them according to the RL algorithm's
        rules.

        Parameters
        ----------
        initial_state: np.ndarray
            Initial state of the environment
        env: BaseEnv
            The environment actions are applied on. Provides the state fed to
            the RL algorithm

        Returns
        -------
        running_reward: float
            Cummulative training steps reward
        actor_loss: float
            Policty gradient loss of actor
        critic_loss: float
            MSE loss of critic
        episode_length: int
            Length of episode aka how many transitions were gathered
        """

        running_reward = 0
        state = initial_state
        done = False
        running_losses = defaultdict(list)

        episode_length = 0
        indices = np.asarray(range(state.shape[0]))

        while not np.all(done):

            # Select action according to policy
            # Noise is already added by the policy
            action = self.policy.select_action(
                state, stochastic=True)

            self.t += action.shape[0]

            v, prob, _, mu, std = self.policy.get_evaluation(
                state,
                action)

            # Perform action
            next_state, reward, done, _ = env.step(action)

            vp, *_ = self.policy.get_evaluation(
                next_state,
                action)

            # Set next state as current state
            running_reward += sum(reward)

            # Store data in replay buffer
            self.replay_buffer.add(
                indices, state.cpu().numpy(), action, next_state.cpu().numpy(),
                reward, done, v, vp, prob, mu, std)

            # "Harvesting" here means removing "done" trajectories
            # from state as well as removing the associated streamlines
            state, idx = env.harvest(next_state)

            indices = indices[idx]

            # Keeping track of episode length
            episode_length += 1

        losses = self.update(
            self.replay_buffer)
        running_losses = add_item_to_means(running_losses, losses)

        return (
            running_reward,
            running_losses,
            episode_length)

    def update(
        self,
        replay_buffer,
        batch_size=4096
    ) -> Tuple[float, float]:
        """
        Policy update function, where we want to maximize the probability of
        good actions and minimize the probability of bad actions

        Therefore:
            - actions with a high probability and positive advantage will
              be made a lot more likely
            - actions with a low probabiliy and positive advantage will be made
              more likely
            - actions with a high probability and negative advantage will be
              made a lot less likely
            - actions with a low probabiliy and negative advantage will be made
              less likely

        Parameters
        ----------
        replay_buffer: ReplayBuffer
            Replay buffer that contains transitions
        batch_size: int
            Batch size to update the actor

        Returns
        -------
        losses: dict
            Dict. containing losses and training-related metrics.
        """

        # Sample replay buffer
        s, a, ret, *_ = \
            replay_buffer.sample()

        running_losses = defaultdict(list)

        for i in range(0, len(s), min(len(s), batch_size)):
            j = i + batch_size
            state = torch.FloatTensor(s[i:j]).to(self.device)
            action = torch.FloatTensor(a[i:j]).to(self.device)
            returns = torch.FloatTensor(ret[i:j]).to(self.device)

            log_prob, entropy, *_ = self.policy.evaluate(state, action)

            # VPG policy loss
            actor_loss = -(log_prob * returns).mean() + \
                -self.entropy_loss_coeff * entropy.mean()

            losses = {'actor_loss': actor_loss.item(),
                      'returns': returns.mean().item(),
                      'entropy': entropy.mean().item()}

            running_losses = add_item_to_means(running_losses, losses)

            # Gradient step
            self.optimizer.zero_grad()
            actor_loss.backward()
            self.optimizer.step()

        return mean_losses(running_losses)
