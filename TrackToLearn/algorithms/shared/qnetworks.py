import math
import numpy as np
import torch
import torch.nn.functional as F

from os.path import join as pjoin
from torch import nn

from TrackToLearn.algorithms.shared.utils import (
    format_widths, make_fc_network)


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    Taken from https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/05.noisy_net.ipynb  # noqa E501

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(
        self, in_features: int, out_features: int, std_init: float = 0.5
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        if self.training:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(
                x,
                self.weight_mu,
                self.bias_mu,
            )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


class QNetwork(nn.Module):
    """
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
    ):
        """
        """

        super(QNetwork, self).__init__()

        self.action_dim = action_dim

        self.hidden_layers = format_widths(hidden_dims)

        self.layers = make_fc_network(
            self.hidden_layers, state_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """ Forward propagation of the q-network.
        Outputs a value for all actions according to the state
        """
        p = self.layers(state)
        return p


class QAgent(nn.Module):
    """
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        device: str
    ):
        """
        """

        super(QAgent, self).__init__()

        self.device = device
        self.action_dim = action_dim
        self.q = QNetwork(
            state_dim, action_dim, hidden_dims).to(self.device)

    def evaluate(self, state) -> torch.Tensor:
        # if state is not batched, expand it to "batch of 1"
        if len(state.shape) < 2:
            state = state[None, :]
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        return self.q(state)

    def act(self, state) -> torch.Tensor:
        return torch.argmax(self.q(state), dim=1)

    def random_action(self, state):
        return np.random.randint(0, self.action_dim, size=state.shape[0])

    def select_action(self, state) -> torch.Tensor:
        """ Select the action which has the highest q-value
        """
        if len(state.shape) < 2:
            state = state[None, :]
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = self.act(state).cpu().data.numpy()

        return action

    def parameters(self):
        """ Access parameters for grad clipping
        """
        return self.q.parameters()

    def load_state_dict(self, state_dict):
        """ Load parameters into actor and critic
        """
        q_state_dict = state_dict
        self.q.load_state_dict(q_state_dict)

    def state_dict(self):
        """ Returns state dicts so they can be loaded into another policy
        """
        return self.q.state_dict()

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
            self.q.state_dict(), pjoin(path, filename + "_q.pth"))

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
        self.q.load_state_dict(
            torch.load(pjoin(path, filename + '_q.pth'),
                       map_location=self.device))

    def eval(self):
        """ Switch actors and critics to eval mode
        """
        self.q.eval()

    def train(self):
        """ Switch actors and critics to train mode
        """
        self.q.train()


def make_shared_fc_network(
    widths, input_size, output_size, activation=nn.ReLU,
    last_activation=nn.Identity
):
    first_layer = [nn.Linear(input_size, widths[0]), activation()]

    layers = []
    for i in range(len(widths[:-1])):
        layers.extend(
            [NoisyLinear(widths[i], widths[i+1]), activation()])

    value_layers = layers.copy()

    layers.extend([NoisyLinear(widths[-1], output_size)])
    value_layers.extend([NoisyLinear(widths[-1], 1)])

    return (
        nn.Sequential(*first_layer),
        nn.Sequential(*layers),
        nn.Sequential(*value_layers))


class DuelingQNetwork(nn.Module):
    """
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
    ):
        """
        """

        super(DuelingQNetwork, self).__init__()

        self.action_dim = action_dim

        self.hidden_layers = format_widths(hidden_dims)

        self.base, self.a, self.v = make_shared_fc_network(
            self.hidden_layers, state_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """ Forward propagation of the q-network.
        Outputs a value for all actions according to the state
        """
        base = self.base(state)
        a = self.a(base)
        v = self.v(base)

        q = v + a - a.mean(dim=-1, keepdim=True)
        return q

    def reset_noise(self):
        """
        """
        for layer in self.a:
            if type(layer) == NoisyLinear:
                layer.reset_noise()
        for layer in self.v:
            if type(layer) == NoisyLinear:
                layer.reset_noise()


class DuelingQAgent(QAgent):
    """
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: str,
        device: str
    ):
        """
        """

        super(DuelingQAgent, self).__init__(
            state_dim, action_dim, hidden_dims, device)

        self.device = device
        self.action_dim = action_dim
        self.q = DuelingQNetwork(
            state_dim, action_dim, hidden_dims).to(self.device)

    def reset_noise(self):
        """
        """
        self.q.reset_noise()
