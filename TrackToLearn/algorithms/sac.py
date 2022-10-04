import copy
import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from nibabel.streamlines import Tractogram
from os.path import join as pjoin
from torch import nn
from torch.distributions.normal import Normal
from typing import Tuple

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.environments.env import BaseEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def add_to_means(means, dic):
    return {k: means[k] + [dic[k]] for k in dic.keys()}


def format_widths(widths_str):
    return [int(i) for i in widths_str.split('-')]


def make_fc_network(
    widths, input_size, output_size, activation=nn.ReLU,
    last_activation=nn.Identity
):
    layers = [nn.Linear(input_size, widths[0]), activation()]
    for i in range(len(widths[:-1])):
        layers.extend(
            [nn.Linear(widths[i], widths[i+1]), activation()])
    # no activ. on last layer
    layers.extend([nn.Linear(widths[-1], output_size)])
    return nn.Sequential(*layers)


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
            self.reward[ind], dtype=torch.float32, device=self.device
        ).squeeze(-1)
        d = torch.as_tensor(
            self.not_done[ind], dtype=torch.float32, device=self.device
        ).squeeze(-1)

        return s, a, ns, r, d

    def clear_memory(self):
        """ Reset the buffer
        """
        self.ptr = 0
        self.size = 0

    def save(self, path, name, i):
        """ TODO for imitation learning
        """
        states_file = pjoin(path, name + "_states_{}.npy".format(i))
        actions_file = pjoin(path, name + "_actions_{}.npy".format(i))
        next_states_file = pjoin(path, name + "_next_states_{}.npy".format(i))
        rewards_file = pjoin(path, name + "_rewards_{}.npy".format(i))
        dones_file = pjoin(path, name + "_dones_{}.npy".format(i))

        np.save(states_file, self.state[:self.size])
        np.save(actions_file, self.action[:self.size])
        np.save(next_states_file, self.next_state[:self.size])
        np.save(rewards_file, self.reward[:self.size])
        np.save(dones_file, 1. - self.not_done[:self.size])

    def load(self, path, i):
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
        hidden_dims: str,
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

        self.action_dim = action_dim

        self.hidden_layers = format_widths(hidden_dims)

        self.layers = make_fc_network(
            self.hidden_layers, state_dim, action_dim * 2)

        self.output_activation = nn.Tanh()

    def forward(
        self,
        state: torch.Tensor,
        stochastic: bool,
        with_logprob: bool = False,
    ) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs an un-noisy un-normalized action
        """

        p = self.layers(state)
        mu = p[:, :self.action_dim]
        log_std = p[:, self.action_dim:]

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)

        if stochastic:
            pi_action = pi_distribution.rsample()
        else:
            pi_action = mu

        # Trick from Spinning Up's implementation:
        # Compute logprob from Gaussian, and then apply correction for Tanh
        # squashing. NOTE: The correction formula is a little bit magic. To
        # get an understanding of where it comes from, check out the
        # original SAC paper (arXiv 1801.01290) and look in appendix C.
        # This is a more numerically-stable equivalent to Eq 21.
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - pi_action -
                       F.softplus(-2*pi_action))).sum(axis=1)

        pi_action = self.output_activation(pi_action)

        return pi_action, logp_pi

    def logprob(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:

        p = self.layers(state)
        mu = p[:, :self.action_dim]
        log_std = p[:, self.action_dim:]

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)

        # Trick from Spinning Up's implementation:
        # Compute logprob from Gaussian, and then apply correction for Tanh
        # squashing. NOTE: The correction formula is a little bit magic. To
        # get an understanding of where it comes from, check out the
        # original SAC paper (arXiv 1801.01290) and look in appendix C.
        # This is a more numerically-stable equivalent to Eq 21.
        logp_pi = pi_distribution.log_prob(action).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - action -
                       F.softplus(-2*action))).sum(axis=1)

        return logp_pi


class Critic(nn.Module):
    """ Critic module that takes in a pair of state-action and outputs its
    q-value according to the network's q function. SAC uses two critics
    and takes the lowest value of the two during backprop.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
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

        self.hidden_layers = format_widths(hidden_dims)

        self.q1 = make_fc_network(
            self.hidden_layers, state_dim + action_dim, 1)
        self.q2 = make_fc_network(
            self.hidden_layers, state_dim + action_dim, 1)

    def forward(self, state, action) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs a q estimate from both critics
        """
        q1_input = torch.cat([state, action], -1)
        q2_input = torch.cat([state, action], -1)

        q1 = self.q1(q1_input).squeeze(-1)
        q2 = self.q2(q2_input).squeeze(-1)

        return q1, q2

    def Q1(self, state, action) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs a q estimate from first critic
        """
        q1_input = torch.cat([state, action], -1)

        q1 = self.q1(q1_input)

        return q1


class ActorCritic(object):
    """ Module that handles the actor and the critic
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: int,
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
            state_dim, action_dim, hidden_dims
        ).to(device)

        self.critic = Critic(
            state_dim, action_dim, hidden_dims
        ).to(device)

    def act(
        self,
        state: torch.Tensor,
        stochastic: bool = True,
    ) -> torch.Tensor:
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
        a, logprob = self.actor(state, stochastic)
        return a, logprob

    def select_action(self, state: np.array, stochastic=True) -> np.ndarray:
        """ Move state to torch tensor, select action and
        move it back to numpy array

        Parameters:
        -----------
            state: np.array
                State of the environment
            deterministic: bool
                Return deterministic action (at test time)

        Returns:
        --------
            action: np.array
                Action selected by the policy
        """
        # if state is not batched, expand it to "batch of 1"
        if len(state.shape) < 2:
            state = state[None, :]

        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        action, _ = self.act(state, stochastic)

        return action.cpu().data.numpy(), None

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


class SAC(RLAlgorithm):
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
        hidden_dims: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        alpha: float = 0.2,
        batch_size: int = 2048,
        gm_seeding: bool = False,
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
        gm_seeding: bool
            If seeding from GM, don't "go back"
        rng: np.random.RandomState
            rng for randomness. Should be fixed with a seed
        device: torch.device,
            Device to use for processing (CPU or GPU)
            Should always on GPU
        """

        super(SAC, self).__init__(
            input_size,
            action_size,
            hidden_dims,
            0.,
            lr,
            gamma,
            batch_size,
            gm_seeding,
            rng,
            device,
        )

        # Initialize main policy
        self.policy = ActorCritic(
            input_size, action_size, hidden_dims,
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

        self.start_timesteps = 0
        self.total_it = 0
        self.policy_freq = 2
        self.tau = 0.005

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

        episode_length = 0

        running_losses = defaultdict(list)

        while not np.all(done):

            # Select action according to policy + noise for exploration
            action, h = self.policy.select_action(
                np.array(state), stochastic=True)

            self.t += action.shape[0]
            # Perform action
            next_state, reward, done, _ = env.step(action)
            done_bool = done

            # Store data in replay buffer
            # WARNING: This is a bit of a trick and I'm not entirely sure this
            # is legal. This is effectively adding to the replay buffer as if
            # I had n agents gathering transitions instead of a single one.
            # This is not mentionned in the SAC paper. PPO2 does use multiple
            # learners, though.
            # I'm keeping it since since it reaaaally speeds up training with
            # no visible costs
            self.replay_buffer.add(
                state, action,
                next_state, reward[..., None],
                done_bool[..., None])

            running_reward += sum(reward)

            # Train agent after collecting sufficient data
            # TODO: Add monitors so that losses are properly tracked
            if self.t >= self.start_timesteps:
                losses = self.update(
                    self.replay_buffer)
                running_losses = add_to_means(running_losses, losses)
            # "Harvesting" here means removing "done" trajectories
            # from state as well as removing the associated streamlines
            # This line also set the next_state as the state
            state, h, _ = env.harvest(next_state, h)

            # Keeping track of episode length
            episode_length += 1

        tractogram = env.get_streamlines()
        return (
            tractogram,
            running_reward,
            running_losses,
            episode_length)

    def update(
        self,
        replay_buffer: ReplayBuffer,
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

        running_actor_loss = 0
        running_critic_loss = 0

        self.total_it += 1

        # Sample replay buffer
        # TODO: Make batch size parametrizable
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
        loss_q1 = ((current_Q1.squeeze(-1) - backup)**2).mean()
        loss_q2 = ((current_Q2.squeeze(-1) - backup)**2).mean()
        critic_loss = loss_q1 + loss_q2

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

        running_actor_loss = actor_loss.detach().cpu().numpy()

        running_critic_loss = critic_loss.detach().cpu().numpy()

        return running_actor_loss, running_critic_loss
