import copy
import numpy as np
import torch

from typing import Tuple

from TrackToLearn.algorithms.ddpg import DDPG
from TrackToLearn.algorithms.shared.offpolicy import SACActorCritic
from TrackToLearn.algorithms.shared.replay import OffPolicyReplayBuffer


class SAC(DDPG):
    """
    The sample-gathering and training algorithm.
    Based on

        Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018, July). Soft
        actor-critic: Off-policy maximum entropy deep reinforcement learning with
        a stochastic actor. In International conference on machine learning
        (pp. 1861-1870). PMLR.

    Implementation is based on Spinning Up's and rlkit

    See https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py
    See https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py  # noqa E501

    Some alterations have been made to the algorithms so it could be
    fitted to the tractography problem.

    """

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_dims: str,
        lr: float = 3e-4,
        gamma: float = 0.99,
        alpha: float = 0.2,
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
        lr: float
            Learning rate for optimizer
        gamma: float
            Gamma parameter future reward discounting
        alpha: float
            Parameter for entropy bonus
        n_actors: int
            Batch size for replay buffer sampling
        rng: np.random.RandomState
            rng for randomness. Should be fixed with a seed
        device: torch.device,
            Device to use for processing (CPU or GPU)
            Should always on GPU
        """

        self.max_action = 1.
        self.t = 1

        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.n_actors = n_actors

        self.rng = rng

        # Initialize main policy
        self.policy = SACActorCritic(
            input_size, action_size, hidden_dims, device,
        )

        # Initialize target policy to provide baseline
        self.target = copy.deepcopy(self.policy)

        # SAC requires a different model for actors and critics
        # Optimizer for actor
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=lr)

        # Optimizer for critic
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=lr)

        # Temperature
        self.alpha = alpha

        # SAC-specific parameters
        self.max_action = 1.
        self.on_policy = False

        self.start_timesteps = 1000
        self.total_it = 0
        self.tau = 0.005

        # Replay buffer
        self.replay_buffer = OffPolicyReplayBuffer(
            input_size, action_size)

        self.rng = rng

    def sample_action(
        self,
        state: torch.Tensor
    ) -> np.ndarray:
        """ Sample an action according to the algorithm.
        """

        # Select action according to policy + noise for exploration
        action = self.policy.select_action(state, stochastic=True)

        return action

    def update(
        self,
        replay_buffer: OffPolicyReplayBuffer,
        batch_size: int = 2**12
    ) -> Tuple[float, float]:
        """

        SAC improves upon DDPG by:
            - Introducing entropy into the objective
            - Using Double Q-Learning to fight overestimation

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

        pi, logp_pi = self.policy.act(state)
        alpha = self.alpha

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
        loss_q1 = ((current_Q1 - backup)**2).mean()
        loss_q2 = ((current_Q2 - backup)**2).mean()
        critic_loss = loss_q1 + loss_q2

        losses = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'loss_q1': loss_q1.item(),
            'loss_q2': loss_q2.item(),
            'Q1': current_Q1.mean().item(),
            'Q2': current_Q2.mean().item(),
            'backup': backup.mean().item(),
        }

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
