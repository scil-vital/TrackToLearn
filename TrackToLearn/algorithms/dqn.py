import numpy as np
import torch
# import torch.nn.functional as F

from collections import defaultdict
from torch.nn.utils import clip_grad_norm_
from typing import Tuple

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.algorithms.shared.qnetworks import DuelingCategoricalAgent
from TrackToLearn.algorithms.shared.per import PrioritizedReplayBuffer
from TrackToLearn.algorithms.shared.replay import OffPolicyReplayBuffer
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
        target_update_freq: int = 200,
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

        self.atoms = 51
        self.v_min = 0.0
        self.v_max = 200.0
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atoms).to(device)

        self.rng = rng

        # Initialize main agent
        self.agent = DuelingCategoricalAgent(
            input_size, action_size, hidden_dims,
            self.atoms, self.support, device,
        )

        # Initialize target agent to provide baseline
        self.target = DuelingCategoricalAgent(
            input_size, action_size, hidden_dims,
            self.atoms, self.support, device,
        )

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
        self.beta_recay = 1.0005
        self.prior_eps = prior_eps

        self.use_per = False

        self.start_timesteps = 1000
        self.total_it = 0
        self.tau = 0.005

        if self.use_per:
            # Prioritized Experience Replay buffer
            self.replay_buffer = PrioritizedReplayBuffer(
                input_size, 1, alpha=self.alpha)
        else:
            # Uniform replay buffer
            self.replay_buffer = OffPolicyReplayBuffer(
                input_size, 1)

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
        # NoisyNets don't require epsilon for exploration
        # if self.epsilon > self.rng.random():
        #     action = self.agent.random_action(state)
        # else:
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

        # PER: increase beta
        if self.use_per:
            # Decay exploration rate
            self.beta = min(1.0, self.beta * self.beta_recay)

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
        batch_size: int = 2 ** 10,
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
        if self.use_per:
            state, action, next_state, reward, not_done, weights, indices = \
                replay_buffer.sample(batch_size, self.beta)
        else:
            state, action, next_state, reward, not_done = \
                replay_buffer.sample(batch_size)

        action = action.to(dtype=torch.int64)

        # with torch.no_grad():
        #     # Compute the target Q value
        #     target_action = self.agent.act(next_state)[..., None]
        #     target_Q = self.target.evaluate(
        #         next_state).gather(1, target_action).squeeze(-1)
        #     backup = reward + not_done * self.gamma * target_Q

        # # Get current Q estimates
        # current_Q = self.agent.evaluate(
        #     state).gather(1, action).squeeze(-1)
        # # Compute Huber loss

        # if self.use_per:
        #     q_loss = F.huber_loss(current_Q, backup, reduction="none")
        #     per_loss = torch.mean(q_loss * weights)
        # else:
        #     q_loss = F.huber_loss(current_Q, backup)

        delta_z = float(self.v_max - self.v_min) / (self.atoms - 1)

        with torch.no_grad():

            next_action = self.agent.evaluate(next_state).argmax(1)
            next_dist = self.target.dist(next_state)
            next_dist = next_dist[range(batch_size), next_action]

            t_z = (reward.unsqueeze(-1) +
                   not_done.unsqueeze(-1) * self.gamma * self.support)
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            print(b)
            # TODO: for loop
            l = b.floor().long()
            u = b.ceil().long()

            # offset = (
            #     torch.linspace(
            #         0, (batch_size - 1) * self.atoms, batch_size
            #     ).long()
            #     .unsqueeze(1)
            #     .expand(batch_size, self.atoms)
            #     .to(self.device)
            # )

            m = torch.zeros(next_dist.size(), device=self.device)
            print(m.size(), l.size(), u.size(), b.size())
            m[l] = m[l] + next_dist * (u.float() - b)
            m[u] = m[u] + next_dist * (b - l.float())
            # proj_dist = torch.zeros(next_dist.size(), device=self.device)
            # proj_dist.view(-1).index_add_(
            #     0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            # )
            # proj_dist.view(-1).index_add_(
            #     0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            # )

        dist = self.agent.dist(state)
        log_p = torch.log(dist[range(batch_size), action])

        q_loss = -(m * log_p).sum(1).mean()

        # Optimize the critic
        self.q_optimizer.zero_grad()
        # if self.use_per:
        #     per_loss.backward()
        # else:
        q_loss.backward()
        clip_grad_norm_(self.agent.parameters(), 10)
        self.q_optimizer.step()

        if self.use_per:
            # PER: update priorities
            loss_for_prior = q_loss.detach().cpu().numpy()
            new_priorities = np.abs(loss_for_prior) + self.prior_eps
            self.replay_buffer.update_priorities(indices, new_priorities)

        # Noisy networks
        self.agent.reset_noise()
        self.target.reset_noise()

        losses = {
            'q_loss': q_loss.mean().item(),
            # 'Q': current_Q.mean().item(),
            # 'Q\'': target_Q.mean().item(),
            'epsilon': self.epsilon,
        }

        # if self.use_per:
        #     losses.update({
        #         'beta': self.beta,
        #         'per_loss': per_loss.item()
        #     })

        # Delayed policy updates
        if self.total_it % self.target_update_freq == 0:

            self.target.load_state_dict(self.agent.state_dict())

        return losses
