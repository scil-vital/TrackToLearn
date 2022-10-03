import numpy as np
import scipy.signal
import torch

from typing import Tuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    """ Replay buffer to store transitions. Efficiency could probably be improved

    TODO: Add possibility to save and load to disk for imitation learning
    """

    def __init__(
        self, state_dim: int, action_dim: int, n_trajectories: int,
        n_updates: int, gamma: float, lmbda: float = 0.95
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
        self.size = 0

        self.n_trajectories = n_trajectories
        self.n_updates = n_updates
        self.device = device
        self.lens = np.zeros((n_trajectories), dtype=np.int)
        self.gamma = gamma
        self.lmbda = lmbda
        self.state_dim = state_dim
        self.action_dim = action_dim

        # RL Buffers "filled with zeros"
        self.state = np.zeros((
            self.n_trajectories, self.n_updates, self.state_dim))
        self.action = np.zeros((
            self.n_trajectories, self.n_updates, self.action_dim))
        self.next_state = np.zeros((
            self.n_trajectories, self.n_updates, self.state_dim))
        self.reward = np.zeros((self.n_trajectories, self.n_updates))
        self.not_done = np.zeros((self.n_trajectories, self.n_updates))
        self.values = np.zeros((self.n_trajectories, self.n_updates))
        self.next_values = np.zeros((self.n_trajectories, self.n_updates))
        self.probs = np.zeros((self.n_trajectories, self.n_updates))
        self.mus = np.zeros(
            (self.n_trajectories, self.n_updates, self.action_dim))
        self.stds = np.zeros(
            (self.n_trajectories, self.n_updates, self.action_dim))

        # GAE buffers
        self.ret = np.zeros((self.n_trajectories, self.n_updates))
        self.adv = np.zeros((self.n_trajectories, self.n_updates))

    def add(
        self,
        ind: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        probs: np.ndarray,
        mus: np.ndarray,
        stds: np.ndarray,
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
        values: np.ndarray
            Batch of "old" value estimates for this batch of transitions
        next_values : np.ndarray
            Batch of "old" value-primes for this batch of transitions
        probs: np.ndarray
            Batch of "old" log-probs for this batch of transitions

        """
        self.state[ind, self.size] = state
        self.action[ind, self.size] = action

        # These are actually not needed
        self.next_state[ind, self.size] = next_state
        self.reward[ind, self.size] = reward
        self.not_done[ind, self.size] = (1. - done)

        # Values for losses
        self.values[ind, self.size] = values
        self.next_values[ind, self.size] = next_values
        self.probs[ind, self.size] = probs

        self.mus[ind, self.size] = mus
        self.stds[ind, self.size] = stds

        self.lens[ind] += 1
        for j in range(len(ind)):
            i = ind[j]

            if done[j] or self.size == self.n_updates - 1:
                # Calculate the expected returns: the value function target
                self.ret[i, :self.size] = scipy.signal.lfilter(
                    [1], [1, -self.gamma], self.reward[i, :self.size][::-1],
                    axis=0)[::-1]

                # Calculate GAE-Lambda with this trick
                # https://stackoverflow.com/a/47971187
                deltas = self.reward[i, :self.size] + \
                    (self.gamma * self.next_values[i, :self.size] *
                     self.not_done[i, :self.size]) - \
                    self.values[i, :self.size]

                if self.lmbda == 0:
                    self.adv[i, :self.size] = self.ret[i, :self.size] - \
                        self.values[i, :self.size]
                else:
                    self.adv[i, :self.size] = scipy.signal.lfilter(
                        [1], [1, -self.gamma * self.lmbda], deltas[::-1],
                        axis=0)[::-1]

        self.size += 1

    def sample(
        self,
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
        ret: torch.Tensor
            Sampled return estimate, target for V
        adv: torch.Tensor
            Sampled advantges, factor for policy update
        probs: torch.Tensor
            Sampled old action probabilities
        """
        # TODO: Not sample whole buffer ? Have M <= N*T

        # Generate indices
        row, col = zip(*((i, l)
                         for i in range(len(self.lens))
                         for l in range(self.lens[i])))

        s = torch.FloatTensor(self.state[row, col]).to(self.device)
        a = torch.FloatTensor(self.action[row, col]).to(self.device)
        ret = torch.FloatTensor(self.ret[row, col]).to(self.device)
        adv = torch.FloatTensor(self.adv[row, col]).to(self.device)
        probs = torch.FloatTensor(self.probs[row, col]).to(self.device)
        mus = torch.FloatTensor(self.mus[row, col]).to(self.device)
        stds = torch.FloatTensor(self.stds[row, col]).to(self.device)

        # Normalize advantage. Needed ?
        # Trick used by OpenAI in their PPO impl
        adv = (adv - adv.mean()) / (adv.std() + 1.e-8)

        return s, a, ret, adv, probs, mus, stds

    def clear_memory(self):
        """ Reset the buffer
        """

        self.lens = np.zeros((self.n_trajectories), dtype=np.int)

        # RL Buffers "filled with zeros"
        self.state = np.zeros((
            self.n_trajectories, self.n_updates, self.state_dim))
        self.action = np.zeros((
            self.n_trajectories, self.n_updates, self.action_dim))
        self.next_state = np.zeros((
            self.n_trajectories, self.n_updates, self.state_dim))
        self.reward = np.zeros((self.n_trajectories, self.n_updates))
        self.not_done = np.zeros((self.n_trajectories, self.n_updates))
        self.values = np.zeros((self.n_trajectories, self.n_updates))
        self.next_values = np.zeros((self.n_trajectories, self.n_updates))
        self.probs = np.zeros((self.n_trajectories, self.n_updates))
        self.mus = np.zeros(
            (self.n_trajectories, self.n_updates, self.action_dim))
        self.stds = np.zeros(
            (self.n_trajectories, self.n_updates, self.action_dim))

        # GAE buffers
        self.ret = np.zeros((self.n_trajectories, self.n_updates))
        self.adv = np.zeros((self.n_trajectories, self.n_updates))

        self.size = 0

    def __len__(self):
        return np.sum(self.lens)

    def save_to_file(self, path):
        """ TODO for imitation learning
        """
        pass

    def load_from_file(self, path):
        """ TODO for imitation learning
        """
        pass


class OffPolicyReplayBuffer(object):
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
