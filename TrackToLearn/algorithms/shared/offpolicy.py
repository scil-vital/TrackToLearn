import numpy as np
import torch
import torch.nn.functional as F

from os.path import join as pjoin
from torch import nn
from torch.distributions.normal import Normal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOG_STD_MAX = 2
LOG_STD_MIN = -20


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
            self.hidden_layers, state_dim, action_dim)

        self.output_activation = nn.Tanh()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """ Forward propagation of the actor.
        Outputs an un-noisy un-normalized action
        """

        return self.output_activation(self.layers(state))


class SACActor(Actor):
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

        q1 = self.q1(q1_input).squeeze(-1)

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
        a = self.actor(state)
        return a

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
        action = self.act(state)

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


class MaxEntropyActorCritic(ActorCritic):
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
        self.actor = SACActor(
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