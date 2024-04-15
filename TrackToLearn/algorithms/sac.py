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
        actor-critic: Off-policy maximum entropy deep reinforcement learning
        with a stochastic actor. In International conference on machine
        learning (pp. 1861-1870). PMLR.

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
        batch_size: int = 2**12,
        replay_size: int = 1e6,
        rng: np.random.RandomState = None,
        device: torch.device = "cuda:0",
    ):
        """ Initialize the algorithm. This includes the replay buffer,
        the policy and the target policy.

        Parameters
        ----------
        input_size: int
            Input size for the model
        action_size: int
            Output size for the actor
        hidden_dims: str
            Dimensions of the hidden layers
        lr: float
            Learning rate for the optimizer(s)
        gamma: float
            Discount factor
        alpha: float
            Entropy regularization coefficient
        n_actors: int
            Number of actors to use
        batch_size: int
            Batch size for the update
        replay_size: int
            Size of the replay buffer
        rng: np.random.RandomState
            Random number generator
        device: torch.device
            Device to train on. Should always be cuda:0
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
        self.agent = SACActorCritic(
            input_size, action_size, hidden_dims, device,
        )

        # Initialize target policy to provide baseline
        self.target = copy.deepcopy(self.agent)

        # SAC requires a different model for actors and critics
        # Optimizer for actor
        self.actor_optimizer = torch.optim.Adam(
            self.agent.actor.parameters(), lr=lr)

        # Optimizer for critic
        self.critic_optimizer = torch.optim.Adam(
            self.agent.critic.parameters(), lr=lr)

        # Temperature
        self.alpha = alpha

        # SAC-specific parameters
        self.max_action = 1.
        self.on_policy = False

        self.start_timesteps = 1000
        self.total_it = 0
        self.tau = 0.005

        self.batch_size = batch_size
        self.replay_size = replay_size

        # Replay buffer
        self.replay_buffer = OffPolicyReplayBuffer(
            input_size, action_size, max_size=replay_size)

        self.rng = rng

    def sample_action(
        self,
        state: torch.Tensor
    ) -> np.ndarray:
        """ Sample an action according to the algorithm.
        """
        # Select action according to policy + noise for exploration
        action = self.agent.select_action(state, probabilistic=1.0)

        return action

    def update(
        self,
        batch,
    ) -> Tuple[float, float]:
        """

        SAC improves over DDPG by introducing an entropy regularization term
        in the actor loss. This encourages the policy to be more stochastic,
        which improves exploration. Additionally, SAC uses the minimum of two
        Q-functions in the value loss, rather than just one Q-function as in
        DDPG. This helps mitigate positive value biases and makes learning more
        stable.

        Parameters
        ----------
        batch: tuple
            Tuple containing the batch of data to train on, including
            state, action, next_state, reward, not_done.

        Returns
        -------
        losses: dict
            Dictionary containing the losses for the actor and critic and
            various other metrics.
        """
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = \
            batch

        pi, logp_pi = self.agent.act(state)
        alpha = self.alpha

        q1, q2 = self.agent.critic(state, pi)
        q_pi = torch.min(q1, q2)

        # Entropy-regularized policy loss
        actor_loss = (alpha * logp_pi - q_pi).mean()

        with torch.no_grad():
            # Target actions come from *current* policy
            next_action, logp_next_action = self.agent.act(next_state)

            # Compute the target Q value
            target_Q1, target_Q2 = self.target.critic(
                next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            backup = reward + self.gamma * not_done * \
                (target_Q - alpha * logp_next_action)

        # Get current Q estimates
        current_Q1, current_Q2 = self.agent.critic(
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
            self.agent.critic.parameters(),
            self.target.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(
            self.agent.actor.parameters(),
            self.target.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return losses
