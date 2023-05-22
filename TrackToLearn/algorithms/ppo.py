import numpy as np
import torch

from collections import defaultdict
from torch import nn
from typing import Tuple

from TrackToLearn.algorithms.a2c import A2C
from TrackToLearn.algorithms.shared.onpolicy import ActorCritic
from TrackToLearn.algorithms.shared.replay import ReplayBuffer
from TrackToLearn.algorithms.shared.utils import (
    add_item_to_means, mean_losses)


# TODO : ADD TYPES AND DESCRIPTION
class PPO(A2C):
    """
    The sample-gathering and training algorithm.
    Based on
        John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford:
            “Proximal Policy Optimization Algorithms”, 2017;
            http://arxiv.org/abs/1707.06347 arXiv:1707.06347

    Implementation is based on
    - https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py # noqa E501
    - https://github.com/seungeunrho/minimalRL/blob/master/ppo-lstm.py
    - https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py # noqa E501

    Some alterations have been made to the algorithms so it could be fitted to the
    tractography problem.

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
        K_epochs: int = 80,
        eps_clip: float = 0.01,
        entropy_loss_coeff: float = 0.01,
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
        K_epochs: int
            How many epochs to run the optimizer using the current samples
            PPO allows for many training runs on the same samples
        max_traj_length: int
            Maximum trajectory length to store in memory.
        eps_clip: float
            Clipping parameter for PPO
        entropy_loss_coeff: float,
            Loss coefficient on policy entropy
            Should sum to 1 with other loss coefficients
        n_actors: int
            Number of learners.
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
            input_size, action_size, hidden_dims, device, action_std,
        ).to(device)

        # Note the optimizer is ran on the target network's params
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr)

        # GAE Parameter
        self.lmbda = lmbda

        # PPO Specific parameters
        self.max_traj_length = max_traj_length
        self.K_epochs = K_epochs
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.entropy_loss_coeff = entropy_loss_coeff

        self.max_action = 1.
        self.t = 1
        self.device = device
        self.n_actors = n_actors

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            input_size, action_size, n_actors,
            max_traj_length, self.gamma, self.lmbda)

        self.rng = rng

    def update(
        self,
        replay_buffer,
        batch_size=4096,
    ) -> Tuple[float, float]:
        """
        Policy update function, where we want to maximize the probability of
        good actions and minimize the probability of bad actions

        The general idea is to compare the current policy and the target
        policies. To do so, the "ratio" is calculated by comparing the
        probabilities of actions for both policies. The ratio is then
        multiplied by the "advantage", which is how better than average
        the policy performs.

        Therefore:
            - actions with a high probability and positive advantage will
              be made a lot more likely
            - actions with a low probabiliy and positive advantage will be made
              more likely
            - actions with a high probability and negative advantage will be
              made a lot less likely
            - actions with a low probabiliy and negative advantage will be made
              less likely

        PPO adds a twist to this where, since the advantage estimation is done
        with your (potentially bad) networks, a "pessimistic view" is used
        where gains will be clamped, so that high gradients (for very probable
        or with a high-amplitude advantage) are tamed. This is to prevent your
        network from diverging too much in the early stages

        Parameters
        ----------
        replay_buffer: ReplayBuffer
            Replay buffer that contains transitions

        Returns
        -------
        losses: dict
            Dict. containing losses and training-related metrics.
        """

        running_losses = defaultdict(list)

        # Sample replay buffer
        s, a, ret, adv, p, *_ = \
            replay_buffer.sample()

        # PPO allows for multiple gradient steps on the same data
        # TODO: Should be switched with the batch ?
        for _ in range(self.K_epochs):

            for i in range(0, len(s), batch_size):
                # you can slice further than an array's length
                j = i + batch_size
                state = torch.FloatTensor(s[i:j]).to(self.device)
                action = torch.FloatTensor(a[i:j]).to(self.device)
                returns = torch.FloatTensor(ret[i:j]).to(self.device)
                advantage = torch.FloatTensor(adv[i:j]).to(self.device)
                old_prob = torch.FloatTensor(p[i:j]).to(self.device)

                # V_pi'(s) and pi'(a|s)
                v, logprob, entropy, *_ = self.policy.evaluate(
                    state,
                    action)

                # Ratio between probabilities of action according to policy and
                # target policies
                assert logprob.size() == old_prob.size(), \
                    '{}, {}'.format(logprob.size(), old_prob.size())
                ratio = torch.exp(logprob - old_prob)

                # Surrogate policy loss
                assert ratio.size() == advantage.size(), \
                    '{}, {}'.format(ratio.size(), advantage.size())

                # Finding V Loss:
                assert returns.size() == v.size(), \
                    '{}, {}'.format(returns.size(), v.size())

                surrogate_policy_loss_1 = ratio * advantage
                surrogate_policy_loss_2 = torch.clamp(
                    ratio,
                    1-self.eps_clip,
                    1+self.eps_clip) * advantage

                # PPO "pessimistic" policy loss
                actor_loss = -(torch.min(
                    surrogate_policy_loss_1,
                    surrogate_policy_loss_2)).mean() + \
                    -self.entropy_loss_coeff * entropy.mean()

                # AC Critic loss
                critic_loss = ((v - returns) ** 2).mean()

                losses = {
                    'actor_loss': actor_loss.item(),
                    'critic_loss': critic_loss.item(),
                    'ratio': ratio.mean().item(),
                    'surrogate_loss_1': surrogate_policy_loss_1.mean().item(),
                    'surrogate_loss_2': surrogate_policy_loss_2.mean().item(),
                    'advantage': advantage.mean().item(),
                    'entropy': entropy.mean().item(),
                    'ret': returns.mean().item(),
                    'v': v.mean().item(),
                }

                running_losses = add_item_to_means(running_losses, losses)

                self.optimizer.zero_grad()
                ((critic_loss * 0.5) + actor_loss).backward()

                # Gradient step
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         0.5)
                self.optimizer.step()

        return mean_losses(running_losses)
