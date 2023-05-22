import numpy as np
import torch

from collections import defaultdict
from typing import Tuple

from TrackToLearn.algorithms.a2c import A2C
from TrackToLearn.algorithms.shared.onpolicy import ActorCritic
from TrackToLearn.algorithms.optim import KFACOptimizer
from TrackToLearn.algorithms.shared.replay import ReplayBuffer
from TrackToLearn.algorithms.shared.utils import (
    add_item_to_means, mean_losses)


# TODO : ADD TYPES AND DESCRIPTION
class ACKTR(A2C):
    """
    The sample-gathering and training algorithm.

        Wu, Y., Mansimov, E., Liao, S., Grosse, R., & Ba, J. (2017).
        Scalable trust-region method for deep reinforcement learning using
        kronecker-factored approximation. arXiv preprint arXiv:1708.05144.

    Implementation is based on
     - https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/a2c_acktr.py # noqa E501
     - https://github.com/alecwangcq/KFAC-Pytorch/blob/master/optimizers/kfac.py

    Some alterations have been made to the algorithms so it could be fitted to the
    tractography problem.

    """

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_dims: int,
        action_std: float = 0.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lmbda: float = 0.99,
        entropy_loss_coeff: float = 0.0001,
        delta: float = 0.001,
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
        delta: float
            Hyperparameter for KFAC. Controls the "distance" between
            the new and old policies.
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

        # Optimizer for actor
        self.optimizer = KFACOptimizer(
            self.policy, lr=lr, kl_clip=delta)

        self.entropy_loss_coeff = entropy_loss_coeff

        # GAE Parameter
        self.lmbda = lmbda

        self.delta = delta

        self.max_traj_length = max_traj_length

        self.max_action = 1.
        self.t = 1
        self.device = device
        self.n_actors = n_actors

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            input_size, action_size, n_actors, self.max_traj_length,
            self.gamma, self.lmbda)

        self.rng = rng

    def update(
        self,
        replay_buffer,
        batch_size: int = 8192,
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

        ACKTR improves upon the standard policy gradient update by computing a
        "trust-region", i.e. a maximum amount the policy can change at each
        update.

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

            # Surrogate policy loss
            assert log_prob.size() == advantage.size(), \
                '{}, {}'.format(log_prob.size(), advantage.size())

            # Finding V Loss:
            assert returns.size() == v.size(), \
                '{}, {}'.format(returns.size(), v.size())

            # Policy loss
            actor_loss = -(log_prob * advantage).mean() + \
                -self.entropy_loss_coeff * entropy.mean()

            # ACKTR critic loss
            # based on ikostrikov's implementation
            critic_loss = ((v - returns) ** 2).mean()

            if self.optimizer.steps % self.optimizer.Ts == 0:
                self.policy.zero_grad()
                pg_fisher_loss = -log_prob.mean()

                noisy_v = v + torch.randn(v.size(), device=self.device)
                vf_fisher_loss = -(v - noisy_v.detach()).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss

                self.optimizer.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                self.optimizer.acc_stats = False

            losses = {'actor_loss': actor_loss.item(),
                      'critic_loss': critic_loss.item(),
                      'v': v.mean().item(),
                      'returns': returns.mean().item(),
                      'adv': advantage.mean().item(),
                      'pg_fisher_loss': pg_fisher_loss.item(),
                      'vf_fisher_loss': vf_fisher_loss.item(),
                      'entropy': entropy.mean().item()}

            running_losses = add_item_to_means(running_losses, losses)

            # Gradient step
            self.optimizer.zero_grad()
            ((critic_loss * 0.5) + actor_loss).backward()
            self.optimizer.step()

        return mean_losses(running_losses)
