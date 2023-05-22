import copy
import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from typing import Tuple

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.algorithms.shared.offpolicy import ActorCritic
from TrackToLearn.algorithms.shared.replay import OffPolicyReplayBuffer
from TrackToLearn.algorithms.shared.utils import add_item_to_means
from TrackToLearn.environments.env import BaseEnv


class DDPG(RLAlgorithm):
    """
    Training algorithm.
    Based on
        Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa,
        Y., ... & Wierstra, D. (2015). Continuous control with deep
        reinforcement learning. arXiv preprint arXiv:1509.02971.

    Implementation is based on
    - https://github.com/sfujim/TD3

    Improvements done by TD3 were removed to get DDPG

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
        n_actors: int
           Number of learners
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

        self.rng = rng

        # Initialize main policy
        self.policy = ActorCritic(
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

        # DDPG-specific parameters
        self.action_std = action_std
        self.max_action = 1.
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
        )

        return action

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

        while not np.all(done):

            # Select action according to policy + noise for exploration
            action = self.sample_action(state)

            self.t += action.shape[0]
            # Perform action
            next_state, reward, done, _ = env.step(action)
            done_bool = done

            # Store data in replay buffer
            # WARNING: This is a bit of a trick and I'm not entirely sure this
            # is legal. This is effectively adding to the replay buffer as if
            # I had n agents gathering transitions instead of a single one.
            # This is not mentionned in the TD3 paper. PPO2 does use multiple
            # learners, though.
            # I'm keeping it since since it reaaaally speeds up training with
            # no visible costs
            self.replay_buffer.add(
                state.cpu().numpy(), action, next_state.cpu().numpy(),
                reward[..., None], done_bool[..., None])

            running_reward += sum(reward)

            # Train agent after collecting sufficient data
            if self.t >= self.start_timesteps:
                losses = self.update(
                    self.replay_buffer)
                running_losses = add_item_to_means(running_losses, losses)

            # "Harvesting" here means removing "done" trajectories
            # from state as well as removing the associated streamlines
            # This line also set the next_state as the state
            state, _ = env.harvest(next_state)

            # Keeping track of episode length
            episode_length += 1

        return (
            running_reward,
            running_losses,
            episode_length)

    def update(
        self,
        replay_buffer: OffPolicyReplayBuffer,
        batch_size: int = 4096
    ) -> Tuple[float, float]:
        """

        DDPG's update rule is quite simple: you can do gradient ascent on the
        critic's q-value for the actor's action and backpropagate through
        the critic into the actor. The critic is updated using a simple MSE
        loss, but noise is added to actions it is judging to make it more
        robust. A target model is used for bootstrapping q-values estimations.
        The target is a polyak average of past models.

        Parameters
        ----------
        replay_buffer: ReplayBuffer
            Replay buffer that contains transitions
        batch_size: int
            Batch size to sample the memory

        Returns
        -------
        losses: dict
            Dict. containing losses and training-related metrics.
        """
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = \
            replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add noise
            noise = torch.randn_like(action) * (self.action_std * 2)
            next_action = self.target.actor(next_state) + noise

            # Compute the target Q value
            target_Q = self.target.critic(
                next_state, next_action)
            target_Q = reward + not_done * self.gamma * target_Q

        # Get current Q estimates
        current_Q = self.policy.critic(
            state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.policy.critic(
            state, self.policy.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        losses = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'Q': current_Q.mean().item(),
            'Q\'': target_Q.mean().item(),
        }

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
