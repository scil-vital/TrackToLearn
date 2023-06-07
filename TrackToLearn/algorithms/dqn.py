import copy
import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from typing import Tuple

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.algorithms.shared.offpolicy import QAgent
from TrackToLearn.algorithms.shared.per import PrioritizedReplayBuffer
from TrackToLearn.algorithms.shared.utils import add_item_to_means
from TrackToLearn.environments.env import BaseEnv


class DQN(RLAlgorithm):
    """
    Training algorithm. While the class is named DQN, it is actually Rainbow,
    or parts of it.

    Based on

    Cite RAINBOW

    Implementation is based on
    - https://github.com/Curt-Park/rainbow-is-all-you-need

    Many thanks to Curt-Park on Github for the modular implementation
    of Rainbow.

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
        epsilon_decay: float = 0.9999,
        target_update_freq: int = 1000,
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
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

        # Initialize main agent
        self.agent = QAgent(
            input_size, action_size, hidden_dims, device,
        )

        # Initialize target agent to provide baseline
        self.target = copy.deepcopy(self.agent)

        # DDPG requires a different model for actors and critics
        # Optimizer for actor
        self.q_optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=lr)

        # DQN-specific parameters
        self.epsilon_decay = epsilon_decay
        self.epsilon = 1.
        self.min_epsilon = 0.1
        self.on_policy = False
        self.target_update_freq = target_update_freq
        self.alpha = alpha
        self.beta = beta
        self.beta_recay = 1.00001
        self.prior_eps = prior_eps

        self.start_timesteps = 1000
        self.total_it = 0
        self.tau = 0.005

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            input_size, 1, alpha=self.alpha)

        self.episode_n = 0
        self.t = 1
        self.rng = rng
        self.device = device
        self.n_actors = n_actors

    def sample_action(
        self,
        state: torch.Tensor
    ) -> np.ndarray:
        """ Epsilon-greedy action sampling
        """
        if self.epsilon > self.rng.random():
            action = self.agent.random_action(state)
        else:
            action = self.agent.select_action(state)
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
        running_reward_factors = defaultdict(list)

        episode_length = 0

        self.episode_n += 1

        # Decay exploration rate
        self.beta = min(1.0, self.beta * self.beta_recay)

        # PER: increase beta
        fraction = min(self.episode_n / 20000000, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)

        while not np.all(done):

            # Select action according to policy + noise for exploration
            action = self.sample_action(state)

            self.t += action.shape[0]
            # Perform action
            next_state, reward, done, info = env.step(action)
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
                state.cpu().numpy(), action[..., None],
                next_state.cpu().numpy(),
                reward[..., None], done_bool[..., None])

            running_reward += sum(reward)

            # Train agent after collecting sufficient data
            if self.t >= self.start_timesteps:
                losses = self.update(
                    self.replay_buffer)
                running_losses = add_item_to_means(running_losses, losses)

                # Decay exploration rate
                self.epsilon = max(self.min_epsilon,
                                   self.epsilon * self.epsilon_decay)

            # "Harvesting" here means removing "done" trajectories
            # from state as well as removing the associated streamlines
            # This line also set the next_state as the state
            state, _ = env.harvest(next_state)

            # Keeping track of episode length
            episode_length += 1
        return (
            running_reward,
            running_losses,
            episode_length,
            running_reward_factors)

    def update(
        self,
        replay_buffer: PrioritizedReplayBuffer,
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
        state, action, next_state, reward, not_done, weights, indices = \
            replay_buffer.sample(batch_size, self.beta)

        action = action.to(dtype=torch.int64)
        with torch.no_grad():
            # Compute the target Q value
            target_action = self.agent.act(next_state)[..., None]
            target_Q = self.target.evaluate(
                next_state).gather(1, target_action)[..., 0]
            backup = reward + not_done * self.gamma * target_Q
        # Get current Q estimates
        current_Q = self.agent.evaluate(
            state).gather(1, action)[..., 0]
        # Compute Huber loss
        q_loss = F.smooth_l1_loss(current_Q, backup, reduction="none")
        per_loss = torch.mean(q_loss * weights)

        # Optimize the critic
        self.q_optimizer.zero_grad()
        per_loss.backward()
        self.q_optimizer.step()

        # PER: update priorities
        loss_for_prior = q_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.replay_buffer.update_priorities(indices, new_priorities)

        losses = {
            'q_loss': q_loss.mean().item(),
            'per_loss': per_loss.item(),
            'Q': current_Q.mean().item(),
            'Q\'': target_Q.mean().item(),
            'epsilon': self.epsilon,
            'beta': self.beta
        }

        # Delayed policy updates
        if self.total_it % self.target_update_freq == 0:

            # Hard update the frozen target models
            for param, target_param in zip(
                self.agent.parameters(),
                self.target.parameters()
            ):
                target_param.data.copy_(param.data)

        return losses
