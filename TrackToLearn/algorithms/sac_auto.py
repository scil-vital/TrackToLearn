import numpy as np
import torch

import torch.nn.functional as F
from typing import Tuple

from TrackToLearn.algorithms.sac import SAC


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SACAuto(SAC):
    """
    The sample-gathering and training algorithm.
    Based on
    TODO: Cite
    Implementation is based on Spinning Up's and rlkit

    See https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py
    See https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py  # noqa E501

    Some alterations have been made to the algorithms so it could be
    fitted to the tractography problem.

    """

    def __init__(
        self,
        input_size: int,
        action_size: int = 3,
        hidden_dims: str = '1024-1024',
        lr: float = 3e-4,
        gamma: float = 0.99,
        alpha: float = 1.0,
        batch_size: int = 2048,
        interface_seeding: bool = False,
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
        hidden_size: int
            Width of the model
        action_std: float
            Standard deviation on actions for exploration
        lr: float
            Learning rate for optimizer
        gamma: float
            Gamma parameter future reward discounting
        interface_seeding: bool
            If seeding from GM, don't "go back"
        rng: np.random.RandomState
            rng for randomness. Should be fixed with a seed
        device: torch.device,
            Device to use for processing (CPU or GPU)
            Should always on GPU
        """

        super(SACAuto, self).__init__(
            input_size,
            action_size,
            hidden_dims,
            lr,
            gamma,
            alpha,
            batch_size,
            interface_seeding,
            rng,
            device,
        )

        # Auto-temperature adjustment
        starting_temperature = np.log(alpha)  # Found empirically
        self.target_entropy = -np.prod(action_size).item()
        self.log_alpha = torch.full(
            (1,), starting_temperature, requires_grad=True, device=device)

        self.alpha_optimizer = torch.optim.Adam(
          [self.log_alpha], lr=lr)

        # SAC-specific parameters
        self.max_action = 1.
        self.on_policy = False

        self.start_timesteps = 1000
        self.total_it = 0
        self.policy_freq = 2
        self.tau = 0.005

        self.rng = rng

    def update(
        self,
        replay_buffer,
        batch_size: int = 2**12
    ) -> Tuple[float, float]:
        """
        TODO: Add motivation behind SAC update ("pessimistic" two-critic
        update, policy that implicitely maximizes the q-function, etc.)

        Parameters
        ----------
        replay_buffer: ReplayBuffer
            Replay buffer that contains transitions
        batch_size: int
            Batch size to sample the memory

        Returns
        -------
        running_actor_loss: float
            Average policy loss over all gradient steps
        running_critic_loss: float
            Average critic loss over all gradient steps
        """
        self.total_it += 1

        # Sample replay buffer
        # TODO: Make batch size parametrizable
        state, action, next_state, reward, not_done = \
            replay_buffer.sample(batch_size)

        pi, logp_pi = self.policy.act(state)
        alpha_loss = -(self.log_alpha * (
            logp_pi + self.target_entropy).detach()).mean()
        alpha = self.log_alpha.exp()

        q1, q2 = self.policy.critic(state, pi)
        q_pi = torch.min(q1, q2)

        # Entropy-regularized policy loss
        actor_loss = (alpha * logp_pi - q_pi).mean()

        with torch.no_grad():
            # Target actions come from *current* policy
            next_action, logp_next_action = self.policy.act(next_state)

            # Compute the target Q value
            target_Q1, target_Q2 = self.target.critic(
                next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            backup = reward + self.gamma * not_done * \
                (target_Q - alpha * logp_next_action)

        # Get current Q estimates
        current_Q1, current_Q2 = self.policy.critic(
            state, action)

        # MSE loss against Bellman backup
        loss_q1 = F.mse_loss(current_Q1, backup.detach()).mean()
        loss_q2 = F.mse_loss(current_Q2, backup.detach()).mean()

        critic_loss = loss_q1 + loss_q2

        losses = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item(),
            'q1': current_Q1.mean().item(),
            'q2': current_Q2.mean().item(),
            'q1_loss': loss_q1.item(),
            'q2_loss': loss_q2.item(),
        }

        # Optimize the temperature
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(
            self.policy.critic.parameters(),
            self.target.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(
            self.policy.actor.parameters(),
            self.target.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return losses
