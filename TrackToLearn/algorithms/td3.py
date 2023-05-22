import copy
import numpy as np
import torch
import torch.nn.functional as F

from typing import Tuple

from TrackToLearn.algorithms.ddpg import DDPG
from TrackToLearn.algorithms.shared.offpolicy import TD3ActorCritic
from TrackToLearn.algorithms.shared.replay import OffPolicyReplayBuffer


class TD3(DDPG):
    """
    The sample-gathering and training algorithm.
    Based on
        Scott Fujimoto, Herke van Hoof, David Meger
        "Addressing Function Approximation Error in
        Actor-Critic Methods", 2018;
        https://arxiv.org/abs/1802.09477 arXiv:1802.09477

    Implementation is based on
    - https://github.com/sfujim/TD3

    Some alterations have been made to the algorithms so it could be
    fitted to the tractography problem.

    """

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_dims: str,
        action_std: float = 0.35,
        lr: float = 3e-4,
        gamma: float = 0.99,
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
        hidden_size: int
            Width of the model
        action_std: float
            Standard deviation on actions for exploration
        lr: float
            Learning rate for optimizer
        gamma: float
            Gamma parameter future reward discounting
        n_actors : int
            Nb. of learners.
        rng: np.random.RandomState
            rng for randomness. Should be fixed with a seed
        device: torch.device,
            Device to use for processing (CPU or GPU)
            Should always on GPU
        """

        self.input_size = input_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma

        # Initialize main policy
        self.policy = TD3ActorCritic(
            input_size, action_size, hidden_dims, device,
        )

        # Initialize target policy to provide baseline
        self.target = copy.deepcopy(self.policy)

        # DDPG requires a different model for actors and critics
        # Optimizer for actor
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=lr)

        # Optimizer for critic
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=lr)

        # TD3-specific parameters
        self.action_std = action_std
        self.max_action = 1.
        self.noise_clip = 1.
        self.policy_freq = 2

        # Off-policy parameters
        self.on_policy = False
        self.start_timesteps = 1000
        self.total_it = 0
        self.tau = 0.005

        # Replay buffer
        self.replay_buffer = OffPolicyReplayBuffer(
            input_size, action_size)

        self.t = 1
        self.rng = rng
        self.device = device
        self.n_actors = n_actors

    def sample_action(
        self,
        state: torch.Tensor
    ) -> np.ndarray:
        """ Sample an action according to the algorithm.
        """

        # Select action according to policy + noise for exploration
        a = self.policy.select_action(state)
        action = (
            a + self.rng.normal(
                0, self.max_action * self.action_std,
                size=a.shape)
        ).clip(-self.max_action, self.max_action)

        return action

    def update(
        self,
        replay_buffer: OffPolicyReplayBuffer,
        batch_size: int = 2**12
    ) -> Tuple[float, float]:
        """
        TD3 improves upon DDPG with three additions:
            - Double Q-Learning to fight overestimation
            - Delaying the update of actors to prevent the "moving target"
              problem
            - Clipping the actions used when estimation q-returns

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
        state, action, next_state, reward, not_done = \
            replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select next action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * (self.action_std * 2)
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (
                self.target.actor(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value for s'
            target_Q1, target_Q2 = self.target.critic(
                next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        # Get current Q estimates for s
        current_Q1, current_Q2 = self.policy.critic(
            state, action)

        # Compute critic loss Q(s,a) - r + yQ(s',a)
        loss_q1 = F.mse_loss(current_Q1, target_Q)
        loss_q2 = F.mse_loss(current_Q2, target_Q)
        critic_loss = loss_q1 + loss_q2

        losses = {
            'actor_loss': 0.0,
            'critic_loss': critic_loss.item(),
            'loss_q1': loss_q1.item(),
            'loss_q2': loss_q2.item(),
            'Q1': current_Q1.mean().item(),
            'Q2': current_Q2.mean().item(),
            'Q\'': target_Q.mean().item(),
        }

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss -Q(s,a)
            actor_loss = -self.policy.critic.Q1(
                state, self.policy.actor(state)).mean()

            losses.update({'actor_loss': actor_loss.item()})

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

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
