import numpy as np
import torch

import torch.nn.functional as F

from typing import Tuple

from TrackToLearn.algorithms.sac_auto import SACAuto
from TrackToLearn.algorithms.shared.offpolicy import CrossQActorCritic
from TrackToLearn.algorithms.shared.replay import OffPolicyReplayBuffer
from TrackToLearn.utils.torch_utils import get_device

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class CrossQ(SACAuto):
    """
    The sample-gathering and training algorithm.
    Based on

        Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., ...
        & Levine, S. (2018). Soft actor-critic algorithms and applications.
        arXiv preprint arXiv:1812.05905.

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
        hidden_dims: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        alpha: float = 0.2,
        n_actors: int = 4096,
        batch_size: int = 2**12,
        replay_size: int = 1e6,
        rng: np.random.RandomState = None,
        device: torch.device = get_device,
    ):
        """
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
            Initial entropy coefficient (temperature).
        n_actors: int
            Number of actors to use
        batch_size: int
            Batch size to sample the memory
        replay_size: int
            Size of the replay buffer
        rng: np.random.RandomState
            Random number generator
        device: torch.device
            Device to use for the algorithm. Should be either "cuda:0"
        """

        self.max_action = 1.
        self.t = 1

        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.n_actors = n_actors

        self.rng = rng

        # Initialize main agent
        self.agent = CrossQActorCritic(
            input_size, action_size, hidden_dims, device,
        )

        self.agent.eval()

        # Auto-temperature adjustment
        # SAC automatically adjusts the temperature to maximize entropy and
        # thus exploration, but reduces it over time to converge to a
        # somewhat deterministic policy.
        starting_temperature = np.log(alpha)  # Found empirically
        self.target_entropy = -np.prod(action_size).item()
        self.log_alpha = torch.full(
            (1,), starting_temperature, requires_grad=True, device=device)
        # Optimizer for alpha
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=lr)

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
        self.on_agent = False

        self.start_timesteps = 80000
        self.total_it = 0
        self.tau = 0.005
        self.agent_freq = 1

        self.batch_size = batch_size
        self.replay_size = replay_size

        # Replay buffer
        self.replay_buffer = OffPolicyReplayBuffer(
            input_size, action_size, max_size=self.replay_size)

        self.rng = rng

    def update(
        self,
        batch,
    ) -> Tuple[float, float]:
        """

        SAC Auto improves upon SAC by automatically adjusting the temperature
        parameter alpha. This is done by optimizing the temperature parameter
        alpha to maximize the entropy of the policy. This is done by
        maximizing the following objective:
            J_alpha = E_pi [log pi(a|s) + alpha H(pi(.|s))]
        where H(pi(.|s)) is the entropy of the policy.


        Parameters
        ----------
        batch: Tuple containing the batch of data to train on.

        Returns
        -------
        losses: dict
            Dictionary containing the losses of the algorithm and various
            other metrics.
        """
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = \
            batch

        self.agent.critic.eval()
        self.agent.actor.eval()

        self.agent.actor.train()
        # Compute \pi_\theta(s_t) and log \pi_\theta(s_t)
        pi, logp_pi = self.agent.act(
            state, probabilistic=1.0)
        self.agent.actor.eval()

        # Compute the temperature loss and the temperature
        alpha_loss = -(self.log_alpha * (
            logp_pi + self.target_entropy).detach()).mean()
        alpha = self.log_alpha.exp()

        # Compute the Q values and the minimum Q value
        q1, q2 = self.agent.critic(state, pi)
        q_pi = torch.min(q1, q2)

        # Entropy-regularized agent loss
        actor_loss = (alpha * logp_pi - q_pi).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Optimize the temperature
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            # Target actions come from *current* agent
            next_action, logp_next_action = self.agent.act(
                next_state, probabilistic=1.0)

        state_cat = torch.cat([state, next_state], dim=0)
        action_cat = torch.cat([action, next_action], dim=0)

        # Compute the next Q values using the target agent
        self.agent.critic.train()
        Q1_w_next, Q2_w_next = self.agent.critic(
            state_cat, action_cat)
        self.agent.critic.eval()

        q1, next_q1 = torch.chunk(Q1_w_next, 2)
        q2, next_q2 = torch.chunk(Q2_w_next, 2)
        target_Q = torch.min(next_q1, next_q2) - alpha * logp_next_action

        # Compute the backup which is the Q-learning "target"
        backup = reward + self.gamma * not_done * \
            (target_Q).detach()

        # MSE loss against Bellman backup
        loss_q1 = F.mse_loss(q1, backup).mean()
        loss_q2 = F.mse_loss(q2, backup).mean()

        # Total critic loss
        critic_loss = loss_q1 + loss_q2

        losses = {
            # 'actor_loss': actor_loss.detach(),
            # 'alpha_loss': alpha_loss.detach(),
            # 'critic_loss': critic_loss.detach(),
            # 'loss_q1': loss_q1.detach(),
            # 'loss_q2': loss_q2.detach(),
            # 'entropy': alpha.detach(),
            # 'Q1': current_Q1.mean().detach(),
            # 'Q2': current_Q2.mean().detach(),
            # 'backup': backup.mean().detach(),
        }

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return losses
