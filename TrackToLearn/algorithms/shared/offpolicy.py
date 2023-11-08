import numpy as np
import torch

import torch.nn.functional as F

from os.path import join as pjoin
from torch import nn
from torch.distributions.normal import Normal

from TrackToLearn.algorithms.shared.utils import (
    format_widths, make_fc_network)


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):
    """ Actor module that takes in a state and outputs an action.
    Its policy is the neural network layers
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        output_activation=nn.Tanh
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dims: str
                String representing layer widths

        """
        super(Actor, self).__init__()

        self.action_dim = action_dim

        self.hidden_layers = format_widths(hidden_dims)

        self.layers = make_fc_network(
            self.hidden_layers, state_dim, action_dim)

        self.output_activation = output_activation()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs an un-noisy un-normalized action
        """
        p = self.layers(state)
        p = self.output_activation(p)

        return p


class MaxEntropyActor(Actor):
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
            hidden_dims: str
                String representing layer widths

        """
        super(MaxEntropyActor, self).__init__(
            state_dim, action_dim, hidden_dims)

        self.action_dim = action_dim

        self.hidden_layers = format_widths(hidden_dims)

        self.layers = make_fc_network(
            self.hidden_layers, state_dim, action_dim * 2)

    def forward(
        self,
        state: torch.Tensor,
        probabilistic: float,
    ) -> torch.Tensor:
        """ Forward propagation of the actor. Log probability is computed
        from the Gaussian distribution of the action and correction
        for the Tanh squashing is applied.

        Parameters:
        -----------
        state: torch.Tensor
            Current state of the environment
        probabilistic: float
            Factor to multiply the standard deviation by when sampling.
            0 means a deterministic policy, 1 means a fully stochastic.
        """
        # Compute mean and log_std from neural network. Instead of
        # have two separate outputs, we have one output of size
        # action_dim * 2. The first action_dim are the means, and
        # the last action_dim are the log_stds.
        p = self.layers(state)
        mu = p[:, :self.action_dim]
        log_std = p[:, self.action_dim:]
        # Constrain log_std inside [LOG_STD_MIN, LOG_STD_MAX]
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        # Compute std from log_std
        std = torch.exp(log_std) * probabilistic
        # Sample from Gaussian distribution using reparametrization trick
        pi_distribution = Normal(mu, std, validate_args=False)
        pi_action = pi_distribution.rsample()

        # Trick from Spinning Up's implementation:
        # Compute logprob from Gaussian, and then apply correction for Tanh
        # squashing. NOTE: The correction formula is a little bit magic. To
        # get an understanding of where it comes from, check out the
        # original SAC paper (arXiv 1801.01290) and look in appendix C.
        # This is a more numerically-stable equivalent to Eq 21.
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        # Squash correction
        logp_pi -= (2*(np.log(2) - pi_action -
                       F.softplus(-2*pi_action))).sum(axis=1)

        # Run actions through tanh to get -1, 1 range
        pi_action = self.output_activation(pi_action)
        # Return action and logprob
        return pi_action, logp_pi


class Critic(nn.Module):
    """ Critic module that takes in a pair of state-action and outputs its
    q-value according to the network's q function.
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
            hidden_dims: int
                String representing layer widths

        """
        super(Critic, self).__init__()

        self.hidden_layers = format_widths(hidden_dims)

        self.q1 = make_fc_network(
            self.hidden_layers, state_dim + action_dim, 1)

    def forward(self, state, action) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs a q estimate from both critics
        """
        q1_input = torch.cat([state, action], -1)

        q1 = self.q1(q1_input).squeeze(-1)

        return q1


class DoubleCritic(Critic):
    """ Critic module that takes in a pair of state-action and outputs its
5   q-value according to the network's q function. TD3 uses two critics
    and takes the lowest value of the two during backprop.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        critic_size_factor=2,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dims: str
                String representing layer widths

        """
        super(DoubleCritic, self).__init__(
            state_dim, action_dim, hidden_dims)

        self.hidden_layers = format_widths(
            hidden_dims) * critic_size_factor

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

        q1 = self.q1(q1_input).squeeze(-1)

        return q1


class ActorCritic(object):
    """ Module that handles the actor and the critic
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        device: torch.device,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dims: int
                String representing layer widths

        """
        self.device = device
        self.actor = Actor(
            state_dim, action_dim, hidden_dims
        ).to(device)

        self.critic = Critic(
            state_dim, action_dim, hidden_dims,
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

    def select_action(self, state: np.array, probabilistic=0.0) -> np.ndarray:
        """ Move state to torch tensor, select action and
        move it back to numpy array

        Parameters:
        -----------
            state: np.array
                State of the environment
            probabilistic: float
                Unused as TD3 does not use probabilistic actions.

        Returns:
        --------
            action: np.array
                Action selected by the policy
        """
        # if state is not batched, expand it to "batch of 1"
        if len(state.shape) < 2:
            state = state[None, :]
        action = self.act(state)

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
            torch.load(pjoin(path, filename + '_critic.pth'),
                       map_location=self.device))
        self.actor.load_state_dict(
            torch.load(pjoin(path, filename + '_actor.pth'),
                       map_location=self.device))

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


class TD3ActorCritic(ActorCritic):
    """ Module that handles the actor and the critic for TD3
    The actor is the same as the DDPG actor, but the critic is different.

    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        device: torch.device,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dims: str
                String representing layer widths
            device: torch.device

        """
        self.device = device
        self.actor = Actor(
            state_dim, action_dim, hidden_dims,
        ).to(device)

        self.critic = DoubleCritic(
            state_dim, action_dim, hidden_dims,
        ).to(device)


class SACActorCritic(ActorCritic):
    """ Module that handles the actor and the critic
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        device: torch.device,
    ):
        """
        Parameters:
        -----------
            state_dim: int
                Size of input state
            action_dim: int
                Size of output action
            hidden_dims: str
                String representing layer widths
            device: torch.device

        """
        self.device = device
        self.actor = MaxEntropyActor(
            state_dim, action_dim, hidden_dims,
        ).to(device)

        self.critic = DoubleCritic(
            state_dim, action_dim, hidden_dims,
        ).to(device)

    def act(self, state: torch.Tensor, probabilistic=1.0) -> torch.Tensor:
        """ Select action according to actor

        Parameters:
        -----------
            state: torch.Tensor
                Current state of the environment
            probabilistic: float
                Factor to multiply the standard deviation by when sampling
                actions.

        Returns:
        --------
            action: torch.Tensor
                Action selected by the policy
            logprob: torch.Tensor
                Log probability of the action
        """
        action, logprob = self.actor(state, probabilistic)
        return action, logprob

    def select_action(self, state: np.array, probabilistic=0.0) -> np.ndarray:
        """ Act on a state and return an action.

        Parameters:
        -----------
            state: np.array
                State of the environment
            probabilistic: float
                Factor to multiply the standard deviation by when sampling
                actions.

        Returns:
        --------
            action: np.array
                Action selected by the policy
        """
        # if state is not batched, expand it to "batch of 1"
        if len(state.shape) < 2:
            state = state[None, :]

        action, _ = self.act(state, probabilistic)

        return action
