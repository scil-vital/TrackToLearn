import copy
import numpy as np
import torch
import torch.nn.functional as F

from nibabel.streamlines import Tractogram
from os.path import join as pjoin
from torch import nn
from typing import Tuple

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.environments.env import BaseEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    """ Replay buffer to store transitions. Implemented in a "ring-buffer"
    fashion. Efficiency could probably be improved

    TODO: Add possibility to save and load to disk for imitation learning
    """

    def __init__(
        self, state_dim: int, action_dim: int, max_size=int(1e6)
    ):
        """
        Parameters:
        -----------
        state_dim: int
            Size of states
        action_dim: int
            Size of actions
        max_size: int
            Number of transitions to store
        """
        self.device = device
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        # Buffers "filled with zeros"
        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros(
            (self.max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray
    ):
        """ Add new transitions to buffer in a "ring buffer" way

        Parameters:
        -----------
        state: np.ndarray
            Batch of states to be added to buffer
        action: np.ndarray
            Batch of actions to be added to buffer
        next_state: np.ndarray
            Batch of next-states to be added to buffer
        reward: np.ndarray
            Batch of rewards obtained for this transition
        done: np.ndarray
            Batch of "done" flags for this batch of transitions
        """

        ind = (np.arange(0, len(state)) + self.ptr) % self.max_size

        self.state[ind] = state
        self.action[ind] = action
        self.next_state[ind] = next_state
        self.reward[ind] = reward
        self.not_done[ind] = 1. - done

        self.ptr = (self.ptr + len(ind)) % self.max_size
        self.size = min(self.size + len(ind), self.max_size)

    def sample(
        self,
        batch_size=1024
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """ Off-policy sampling. Will sample min(batch_size, self.size)
        transitions in an unordered way. This removes the ability to do
        GAE and reward discounting after the transitions are sampled

        Parameters:
        -----------
        batch_size: int
            Number of transitions to sample

        Returns:
        --------
        s: torch.Tensor
            Sampled states
        a: torch.Tensor
            Sampled actions
        ns: torch.Tensor
            Sampled s'
        r: torch.Tensor
            Sampled non-discounted rewards
        d: torch.Tensor
            Sampled 1-done flags
        """

        ind = np.random.randint(0, self.size, size=int(batch_size))

        s = torch.as_tensor(
            self.state[ind], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(
            self.action[ind], dtype=torch.float32, device=self.device)
        ns = \
            torch.as_tensor(
                self.next_state[ind], dtype=torch.float32, device=self.device)
        r = torch.as_tensor(
            self.reward[ind], dtype=torch.float32, device=self.device)
        d = torch.as_tensor(
            self.not_done[ind], dtype=torch.float32, device=self.device)

        return s, a, ns, r, d

    def clear_memory(self):
        """ Reset the buffer
        """
        self.ptr = 0
        self.size = 0

    def save_to_file(self, path):
        """ TODO for imitation learning
        """
        pass

    def load_from_file(self, path):
        """ TODO for imitation learning
        """
        pass


class Actor(nn.Module):
    """ Actor module that takes in a state and outputs an action.
    Its policy is the neural network layers
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int = 3,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dim: int
                Width of network. Presumes all intermediary
                layers are of same size for simplicity

        """
        super(Actor, self).__init__()

        self.hidden_layers = hidden_layers
        self.output_activation = nn.Tanh()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs an un-noisy un-normalized action
        """
        p = state
        p = torch.relu(self.l1(p))
        p = torch.relu(self.l2(p))
        p = self.output_activation(self.l3(p))

        return p


class Critic(nn.Module):
    """ Critic module that takes in a pair of state-action and outputs its
5   q-value according to the network's q function. TD3 uses two critics
    and takes the lowest value of the two during backprop.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int = 3,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dim: int
                Width of network. Presumes all intermediary
                layers are of same size for simplicity

        """
        super(Critic, self).__init__()

        self.hidden_layers = hidden_layers
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs a q estimate from both critics
        """
        q1 = torch.cat([state, action], -1)
        q2 = torch.cat([state, action], -1)

        q1 = torch.relu(self.l1(q1))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.relu(self.l4(q2))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs a q estimate from first critic
        """
        q1 = torch.cat([state, action], -1)

        q1 = torch.relu(self.l1(q1))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class ActorCritic(object):
    """ Module that handles the actor and the critic
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int = 3,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dim: int
                Width of network. Presumes all intermediary
                layers are of same size for simplicity

        """
        self.actor = Actor(
            state_dim, action_dim, hidden_dim, hidden_layers,
        ).to(device)

        self.critic = Critic(
            state_dim, action_dim, hidden_dim, hidden_layers,
        ).to(device)

    def act(self, state: torch.Tensor) -> torch.Tensor:
        """ Select action according to actor

        Parameters:
        -----------
            state: torch.Tensor
                Current state of the environment

        Returns:
        --------
            action: torch.Tensor
                Action selected by the policy
        """
        return self.actor(state)

    def select_action(self, state: np.array, stochastic=False) -> np.ndarray:
        """ Move state to torch tensor, select action and
        move it back to numpy array

        Parameters:
        -----------
            state: np.array
                State of the environment

        Returns:
        --------
            action: np.array
                Action selected by the policy
        """
        # if state is not batched, expand it to "batch of 1"
        if len(state.shape) < 2:
            state = state[None, :]

        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        action = self.act(state).cpu().data.numpy()

        return action

    def parameters(self):
        """ Access parameters for grad clipping
        """
        return self.actor.parameters()

    def load_state_dict(self, state_dict):
        """ Load parameters into actor and critic
        """
        actor_state_dict, critic_state_dict = state_dict
        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)

    def state_dict(self):
        """ Returns state dicts so they can be loaded into another policy
        """
        return self.actor.state_dict(), self.critic.state_dict()

    def save(self, path: str, filename: str):
        """ Save policy at specified path and filename
        Parameters:
        -----------
            path: string
                Path to folder that will contain saved state dicts
            filename: string
                Name of saved models. Suffixes for actors and critics
                will be appended
        """
        torch.save(
            self.critic.state_dict(), pjoin(path, filename + "_critic.pth"))
        torch.save(
            self.actor.state_dict(), pjoin(path, filename + "_actor.pth"))

    def load(self, path: str, filename: str):
        """ Load policy from specified path and filename
        Parameters:
        -----------
            path: string
                Path to folder containing saved state dicts
            filename: string
                Name of saved models. Suffixes for actors and critics
                will be appended
        """
        self.critic.load_state_dict(
            torch.load(pjoin(path, filename + '_critic.pth')))
        self.actor.load_state_dict(
            torch.load(pjoin(path, filename + '_actor.pth')))

    def eval(self):
        """ Switch actors and critics to eval mode
        """
        self.actor.eval()
        self.critic.eval()

    def train(self):
        """ Switch actors and critics to train mode
        """
        self.actor.train()
        self.critic.train()


class TD3(RLAlgorithm):
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
        action_size: int = 3,
        hidden_size: int = 256,
        hidden_layers: int = 3,
        action_std: float = 0.35,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 2048,
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
        batch_size: int
            Batch size for replay buffer sampling
        rng: np.random.RandomState
            rng for randomness. Should be fixed with a seed
        device: torch.device,
            Device to use for processing (CPU or GPU)
            Should always on GPU
        """

        super(TD3, self).__init__(
            input_size,
            action_size,
            hidden_size,
            action_std,
            lr,
            gamma,
            batch_size,
            rng,
            device,
        )

        # Initialize main policy
        self.policy = ActorCritic(
            input_size, action_size, hidden_size, hidden_layers,
        )

        # Initialize target policy to provide baseline
        self.target = copy.deepcopy(self.policy)

        # TD3 requires a different model for actors and critics
        # Optimizer for actor
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=lr)

        # Optimizer for critic
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=lr)

        # TD3-specific parameters
        self.max_action = 1.
        self.on_policy = False

        self.start_timesteps = 1000
        self.total_it = 0
        self.policy_freq = 2
        self.tau = 0.005
        self.noise_clip = 1

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            input_size, action_size)

        self.rng = rng

    def _episode(
        self,
        initial_state: np.ndarray,
        env: BaseEnv,
    ) -> Tuple[Tractogram, float, float, float, int]:
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
        tractogram: Tractogram
            Tractogram containing the tracked streamline
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
        tractogram = None
        done = False
        actor_loss = 0
        critic_loss = 0

        episode_length = 0

        while not np.all(done):

            # Select action according to policy + noise for exploration
            a = self.policy.select_action(np.array(state))
            action = (
                a + self.rng.normal(
                    0, self.max_action * self.action_std,
                    size=a.shape)
            ).clip(-self.max_action, self.max_action)

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
                state, action, next_state,
                reward[..., None], done_bool[..., None])

            running_reward += sum(reward)

            # Train agent after collecting sufficient data
            # TODO: Add monitors so that losses are properly tracked
            if self.t >= self.start_timesteps:
                actor_loss, critic_loss = self.update(
                    self.replay_buffer)

            # "Harvesting" here means removing "done" trajectories
            # from state as well as removing the associated streamlines
            # This line also set the next_state as the state
            new_tractogram, state, _ = env.harvest(next_state)

            # Add streamlines to the lot
            if len(new_tractogram.streamlines) > 0:
                if tractogram is None:
                    tractogram = new_tractogram
                else:
                    tractogram += new_tractogram

            # Keeping track of episode length
            episode_length += 1

        return (
            tractogram,
            running_reward,
            actor_loss,
            critic_loss,
            episode_length)

    def update(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 2**12
    ) -> Tuple[float, float]:
        """
        TODO: Add motivation behind TD3 update ("pessimistic" two-critic
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

        running_actor_loss = 0
        running_critic_loss = 0

        self.total_it += 1

        # Sample replay buffer
        # TODO: Make batch size parametrizable
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
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss -Q(s,a)
            actor_loss = -self.policy.critic.Q1(
                state, self.policy.actor(state)).mean()

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

            running_actor_loss = actor_loss.detach().cpu().numpy()

        running_critic_loss = critic_loss.detach().cpu().numpy()

        torch.cuda.empty_cache()

        return running_actor_loss, running_critic_loss
