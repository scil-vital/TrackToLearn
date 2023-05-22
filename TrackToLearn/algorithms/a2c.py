import numpy as np
import torch

from collections import defaultdict
from torch import nn
from typing import Tuple

from TrackToLearn.algorithms.vpg import VPG
from TrackToLearn.algorithms.shared.onpolicy import ActorCritic
from TrackToLearn.algorithms.shared.replay import ReplayBuffer
from TrackToLearn.algorithms.shared.utils import (
    add_item_to_means, mean_losses)


class A2C(VPG):
    """
    The sample-gathering and training algorithm.

    Based on
        Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T.,
        ... & Kavukcuoglu, K. (2016, June). Asynchronous methods for deep
        reinforcement learning. In International conference on machine learning
        (pp. 1928-1937). PMLR.

    Implementation is based on these PPO implementations
    - https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py # noqa E501
    - https://github.com/seungeunrho/minimalRL/blob/master/ppo-lstm.py
    - https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py # noqa E501

    and PPO-specific parts were removed to obtain a simple actor-critic algorithm.
    """

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_dims: str,
        action_std: float = 0.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lmbda: float = 0.99,
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
        lmbda: float
            Lambda parameter for Generalized Advantage Estimation (GAE):
            John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan:
            “High-Dimensional Continuous Control Using Generalized
             Advantage Estimation”, 2015;
            http://arxiv.org/abs/1506.02438 arXiv:1506.02438
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

        self.on_policy = True

        # Declare policy
        self.policy = ActorCritic(
            input_size, action_size, hidden_dims, device, action_std
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr)

        self.entropy_loss_coeff = entropy_loss_coeff

        # GAE Parameter
        self.lmbda = lmbda

        self.max_traj_length = max_traj_length

        self.max_action = 1.
        self.t = 1
        self.device = device
        self.n_actors = n_actors

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            input_size, action_size, self.n_actors, self.max_traj_length,
            self.gamma, self.lmbda)

        self.rng = rng

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

        Returns
        -------
        losses: dict
            Dict. containing losses and training-related metrics.
        """

        # Sample replay buffer
        s, a, ret, adv, *_ = \
            replay_buffer.sample()

        running_losses = defaultdict(list)

        for i in range(0, len(s), batch_size):
            j = i + batch_size

            state = torch.FloatTensor(s[i:j]).to(self.device)
            action = torch.FloatTensor(a[i:j]).to(self.device)
            returns = torch.FloatTensor(ret[i:j]).to(self.device)
            advantage = torch.FloatTensor(adv[i:j]).to(self.device)

            v, log_prob, entropy, *_ = self.policy.evaluate(state, action)

            # assert log_prob.size() == returns.size(), \
            #     '{}, {}'.format(log_prob.size(), returns.size())

            # VPG policy loss
            actor_loss = -(log_prob * advantage).mean() + \
                -self.entropy_loss_coeff * entropy.mean()

            # AC Critic loss
            critic_loss = ((v - returns) ** 2).mean()

            losses = {'actor_loss': actor_loss.item(),
                      'critic_loss': critic_loss.item(),
                      'entropy': entropy.mean().item(),
                      'v': v.mean().item(),
                      'returns': returns.mean().item(),
                      'adv': advantage.mean().item()}

            running_losses = add_item_to_means(running_losses, losses)

            self.optimizer.zero_grad()
            ((critic_loss * 0.5) + actor_loss).backward()

            # Gradient step
            nn.utils.clip_grad_norm_(self.policy.parameters(),
                                     0.5)
            self.optimizer.step()

        return mean_losses(running_losses)
