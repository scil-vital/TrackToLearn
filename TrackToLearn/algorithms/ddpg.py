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
from TrackToLearn.utils.torch_utils import get_device

class DDPG(RLAlgorithm):
    """
    NOTE: LEGACY CODE. The `_episode` function is used. The actual DDPG
    learning algorithm has not been tested in a while.

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
        batch_size: int = 2**12,
        replay_size: int = 1e6,
        rng: np.random.RandomState = None,
        device: torch.device = get_device(),
    ):
        """
        Parameters
        ----------
        input_size: int
            Input size for the model
        action_size: int
            Output size for the actor
        hidden_dims: str
            Dimensions of the hidden layers for the actor and critic
        action_std: float
            Standard deviation of the noise added to the actor's output
        lr: float
            Learning rate for the optimizer(s)
        gamma: float
            Discount factor
        n_actors: int
            Number of actors to use
        batch_size: int
            Batch size to sample the replay buffer
        replay_size: int
            Size of the replay buffer
        rng: np.random.RandomState
            Random number generator
        device: torch.device
            Device to train on. Should always be cuda:0
        """

        self.input_size = input_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma

        self.rng = rng

        # Initialize main policy
        self.agent = ActorCritic(
            input_size, action_size, hidden_dims, device,
        )

        # Initialize target policy to provide baseline
        self.target = copy.deepcopy(self.agent)

        # DDPG requires a different model for actors and critics
        # Optimizer for actor
        self.actor_optimizer = torch.optim.Adam(
            self.agent.actor.parameters(), lr=lr)

        # Optimizer for critic
        self.critic_optimizer = torch.optim.Adam(
            self.agent.critic.parameters(), lr=lr)

        # DDPG-specific parameters
        self.action_std = action_std
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

        self.t = 1
        self.rng = rng
        self.device = device
        self.n_actors = n_actors

    def sample_action(
        self,
        state: torch.Tensor
    ) -> np.ndarray:
        """ Sample an action according to the algorithm.
        DDPG uses a deterministic policy, so no noise is added to the action
        to explore.
        """

        with torch.no_grad():
            # Select action according to policy + noise for exploration
            a = self.agent.select_action(state)
            action = (
                a + torch.normal(
                    0, self.max_action * self.action_std,
                    size=a.shape, device=self.device)
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
            Sum of rewards gathered during the episode
        running_losses: dict
            Dict. containing losses and training-related metrics.
        episode_length: int
            Length of the episode
        running_reward_factors: dict
            Dict. containing the factors that contributed to the reward
        """

        running_reward = 0
        state = initial_state
        done = False
        running_losses = defaultdict(list)
        running_reward_factors = defaultdict(list)

        episode_length = 0

        while not np.all(done):

            # Select action according to policy + noise for exploration
            with torch.no_grad():
                action = self.sample_action(state)

                # Perform action
                next_state, reward, done, info = env.step(
                    action.to(device='cpu', copy=True).numpy())
                done_bool = done

                running_reward_factors = add_item_to_means(
                    running_reward_factors, info['reward_info'])

                # Store data in replay buffer
                # WARNING: This is a bit of a trick and I'm not entirely sure this
                # is legal. This is effectively adding to the replay buffer as if
                # I had n agents gathering transitions instead of a single one.
                # This is not mentionned in the TD3 paper. PPO2 does use multiple
                # learners, though.
                # I'm keeping it since since it reaaaally speeds up training with
                # no visible costs
                self.replay_buffer.add(
                    state.to('cpu', copy=True),
                    action.to('cpu', copy=True),
                    next_state.to('cpu', copy=True),
                    torch.as_tensor(reward[..., None], dtype=torch.float32),
                    torch.as_tensor(done_bool[..., None], dtype=torch.float32))

                running_reward += sum(reward)

            # Train agent after collecting sufficient data
            if self.t >= self.start_timesteps:

                batch = self.replay_buffer.sample(self.batch_size)
                losses = self.update(
                    batch)
                running_losses = add_item_to_means(running_losses, losses)

            self.t += action.shape[0]
            with torch.no_grad():
                # "Harvesting" here means removing "done" trajectories
                # from state as well as removing the associated streamlines
                # This line also set the next_state as the state
                state = env.harvest()

            # Keeping track of episode length
            episode_length += 1
        return (
            running_reward,
            running_losses,
            episode_length,
            running_reward_factors)

    def update(
        self,
        batch,
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
        batch: tuple
            Tuple containing the batch of data to train on, including state,
            action, next_state, reward, not_done.

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

        with torch.no_grad():
            # Select action according to policy and add noise
            noise = torch.randn_like(action) * (self.action_std * 2)
            next_action = self.target.actor(next_state) + noise

            # Compute the target Q value using the target critic
            target_Q = self.target.critic(
                next_state, next_action)
            target_Q = reward + not_done * self.gamma * target_Q

        # Get current Q estimates
        current_Q = self.agent.critic(
            state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.agent.critic(
            state, self.agent.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        losses = {
            'actor_loss': actor_loss.detach(),
            'critic_loss': critic_loss.detach(),
            'Q': current_Q.mean().detach(),
            'Q\'': target_Q.mean().detach(),
        }

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
