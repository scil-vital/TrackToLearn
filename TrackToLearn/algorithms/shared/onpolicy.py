import numpy as np
import torch

from os.path import join as pjoin
from torch import nn
from torch.distributions.normal import Normal
from typing import Tuple

from TrackToLearn.algorithms.shared.utils import (
    format_widths, make_fc_network)


class Actor(nn.Module):
    """ Actor module that takes in a state and outputs an action.
    Its policy is the neural network layers
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        device: torch.device,
        action_std: float = 0.0,
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

        self.hidden_layers = format_widths(hidden_dims)

        self.layers = make_fc_network(
            self.hidden_layers, state_dim, action_dim, activation=nn.Tanh)

        # State-independent STD, as opposed to SAC which uses a
        # state-dependent STD.
        # See https://spinningup.openai.com/en/latest/algorithms/sac.html
        # in the "You Should Know" box
        log_std = -action_std * np.ones(action_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))

    def _mu(self, state: torch.Tensor):
        return self.layers(state)

    def _distribution(self, state: torch.Tensor):
        mu = self._mu(state)
        std = torch.exp(self.log_std)
        try:
            dist = Normal(mu, std)
        except ValueError as e:
            print(mu, std)
            raise e

        return dist

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs an un-noisy un-normalized action
        """
        return self._distribution(state)


class PolicyGradient(nn.Module):
    """ PolicyGradient module that handles actions
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        device: torch.device,
        action_std: float = 0.0,
    ):
        super(PolicyGradient, self).__init__()
        self.device = device
        self.action_dim = action_dim

        self.actor = Actor(
            state_dim, action_dim, hidden_dims, action_std,
        ).to(device)

    def act(
        self, state: torch.Tensor, stochastic: bool = True,
    ) -> torch.Tensor:
        """ Select noisy action according to actor
        """
        pi = self.actor.forward(state)
        # Should always be stochastic
        if stochastic:
            action = pi.sample()  # if stochastic else pi.mean
        else:
            action = pi.mean

        return action

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Get output of value function for the actions, as well as
        logprob of actions and entropy of policy for loss
        """

        pi = self.actor(state)
        mu, std = pi.mean, pi.stddev
        action_logprob = pi.log_prob(action).sum(axis=-1)
        entropy = pi.entropy()

        return action_logprob, entropy, mu, std

    def select_action(
        self, state: np.array, stochastic=True,
    ) -> np.ndarray:
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

        if len(state.shape) < 2:
            state = state[None, :]

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = self.act(state, stochastic).cpu().data.numpy()

        return action

    def get_evaluation(
        self, state: np.array, action: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """ Move state and action to torch tensor,
        get value estimates for states, probabilities of actions
        and entropy for action distribution, then move everything
        back to numpy array

        Parameters:
        -----------
            state: np.array
                State of the environment
            action: np.array
                Actions taken by the policy

        Returns:
        --------
            v: np.array
                Value estimates for state
            prob: np.array
                Probabilities of actions
            entropy: np.array
                Entropy of policy
        """

        if len(state.shape) < 2:
            state = state[None, :]
        if len(action.shape) < 2:
            action = action[None, :]

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(
            action, dtype=torch.float32, device=self.device)

        prob, entropy, mu, std = self.evaluate(state, action)

        # REINFORCE does not use a critic
        values = np.zeros((state.size()[0]))

        return (
            values,
            prob.cpu().data.numpy(),
            entropy.cpu().data.numpy(),
            mu.cpu().data.numpy(),
            std.cpu().data.numpy())

    def load_state_dict(self, state_dict):
        """ Load parameters into actor and critic
        """
        actor_state_dict = state_dict
        self.actor.load_state_dict(actor_state_dict)

    def state_dict(self):
        """ Returns state dicts so they can be loaded into another policy
        """
        return self.actor.state_dict()

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
        self.actor.load_state_dict(
            torch.load(pjoin(path, filename + '_actor.pth'),
                       map_location=self.device))

    def eval(self):
        """ Switch actors and critics to eval mode
        """
        self.actor.eval()

    def train(self):
        """ Switch actors and critics to train mode
        """
        self.actor.train()


class Critic(nn.Module):
    """ Critic module that takes in a pair of state-action and outputs its
    q-value according to the network's q function. TD3 uses two critics
    and takes the lowest value of the two during backprop.
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
        super(Critic, self).__init__()

        self.hidden_layers = format_widths(hidden_dims)

        self.layers = make_fc_network(
            self.hidden_layers, state_dim, 1, activation=nn.Tanh)

    def forward(self, state) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs a q estimate from first critic
        """

        return self.layers(state)


class ActorCritic(PolicyGradient):
    """ Actor-Critic module that handles both actions and values
    Actors and critics here don't share a body but do share a loss
    function. Therefore they are both in the same module
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        device: torch.device,
        action_std: float = 0.0,
    ):
        super(ActorCritic, self).__init__(
            state_dim,
            action_dim,
            hidden_dims,
            device,
            action_std
        )

        self.critic = Critic(
            state_dim, action_dim, hidden_dims,
        ).to(self.device)

        print(self)

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Get output of value function for the actions, as well as
        logprob of actions and entropy of policy for loss
        """

        pi = self.actor.forward(state)
        mu, std = pi.mean, pi.stddev
        action_logprob = pi.log_prob(action).sum(axis=-1)
        entropy = pi.entropy()
        values = self.critic(state).squeeze(-1)

        return values, action_logprob, entropy, mu, std

    def get_evaluation(
        self, state: np.array, action: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """ Move state and action to torch tensor,
        get value estimates for states, probabilities of actions
        and entropy for action distribution, then move everything
        back to numpy array

        Parameters:
        -----------
            state: np.array
                State of the environment
            action: np.array
                Actions taken by the policy

        Returns:
        --------
            v: np.array
                Value estimates for state
            prob: np.array
                Probabilities of actions
            entropy: np.array
                Entropy of policy
        """

        if len(state.shape) < 2:
            state = state[None, :]
        if len(action.shape) < 2:
            action = action[None, :]


        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(
            action, dtype=torch.float32, device=self.device)

        v, prob, entropy, mu, std = self.evaluate(state, action)

        return (
            v.cpu().data.numpy(),
            prob.cpu().data.numpy(),
            entropy.cpu().data.numpy(),
            mu.cpu().data.numpy(),
            std.cpu().data.numpy())

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
