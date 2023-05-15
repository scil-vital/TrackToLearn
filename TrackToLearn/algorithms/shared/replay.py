import numpy as np
import scipy.signal
import torch

from typing import Tuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    """ Replay buffer to store transitions. Efficiency could probably be
    improved.

    While it is called a ReplayBuffer, it is not actually one as no "Replay"
    is performed. As it is used by on-policy algorithms, the buffer should
    be cleared every time it is sampled.

    TODO: Add possibility to save and load to disk for imitation learning
    """

    def __init__(
        self, state_dim: int, action_dim: int, n_trajectories: int,
        max_traj_length: int, gamma: float, lmbda: float = 0.95
    ):
        """
        Parameters:
        -----------
        state_dim: int
            Size of states
        action_dim: int
            Size of actions
        n_trajectories: int
            Number of learned accumulating transitions
        max_traj_length: int
            Maximum length of trajectories
        gamma: float
            Discount factor.
        lmbda: float
            GAE factor.
        """
        self.ptr = 0

        self.n_trajectories = n_trajectories
        self.max_traj_length = max_traj_length
        self.device = device
        self.lens = np.zeros((n_trajectories), dtype=np.int32)
        self.gamma = gamma
        self.lmbda = lmbda
        self.state_dim = state_dim
        self.action_dim = action_dim

        # RL Buffers "filled with zeros"
        self.state = np.zeros((
            self.n_trajectories, self.max_traj_length, self.state_dim))
        self.action = np.zeros((
            self.n_trajectories, self.max_traj_length, self.action_dim))
        self.next_state = np.zeros((
            self.n_trajectories, self.max_traj_length, self.state_dim))
        self.reward = np.zeros((self.n_trajectories, self.max_traj_length))
        self.not_done = np.zeros((self.n_trajectories, self.max_traj_length))
        self.values = np.zeros((self.n_trajectories, self.max_traj_length))
        self.next_values = np.zeros(
            (self.n_trajectories, self.max_traj_length))
        self.probs = np.zeros((self.n_trajectories, self.max_traj_length))
        self.mus = np.zeros(
            (self.n_trajectories, self.max_traj_length, self.action_dim))
        self.stds = np.zeros(
            (self.n_trajectories, self.max_traj_length, self.action_dim))

        # GAE buffers
        self.ret = np.zeros((self.n_trajectories, self.max_traj_length))
        self.adv = np.zeros((self.n_trajectories, self.max_traj_length))

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
        self.state[ind, self.ptr] = state
        self.action[ind, self.ptr] = action

        # These are actually not needed
        self.next_state[ind, self.ptr] = next_state
        self.reward[ind, self.ptr] = reward
        self.not_done[ind, self.ptr] = (1. - done)

        # Values for losses
        self.values[ind, self.ptr] = values
        self.next_values[ind, self.ptr] = next_values
        self.probs[ind, self.ptr] = probs

        self.mus[ind, self.ptr] = mus
        self.stds[ind, self.ptr] = stds

        self.lens[ind] += 1

        for j in range(len(ind)):
            i = ind[j]

            if done[j]:
                # Calculate the expected returns: the value function target
                rew = self.reward[i, :self.ptr]
                # rew = (rew - rew.mean()) / (rew.std() + 1.e-8)
                self.ret[i, :self.ptr] = \
                    self.discount_cumsum(
                        rew, self.gamma)

                # Calculate GAE-Lambda with this trick
                # https://stackoverflow.com/a/47971187
                # TODO: make sure that this is actually correct
                # TODO?: do it the usual way with a backwards loop
                deltas = rew + \
                    (self.gamma * self.next_values[i, :self.ptr] *
                     self.not_done[i, :self.ptr]) - \
                    self.values[i, :self.ptr]

                if self.lmbda == 0:
                    self.adv[i, :self.ptr] = self.ret[i, :self.ptr] - \
                        self.values[i, :self.ptr]
                else:
                    self.adv[i, :self.ptr] = \
                        self.discount_cumsum(deltas, self.gamma * self.lmbda)

        self.ptr += 1

    def discount_cumsum(self, x, discount):
        """
        # Taken from spinup implementation
        magic from rllab for computing discounted cumulative sums of vectors.
        input:
                vector x,
                [x0,
                 x1,
                 x2]
        output:
                [x0 + discount * x1 + discount^2 * x2,
                 x1 + discount * x2,
                 x2]
        """
        return scipy.signal.lfilter(
            [1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def sample(
        self,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """ Sample all transitions.

        Parameters:
        -----------

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
        # TODO?: Not sample whole buffer ? Have M <= N*T ?

        # Generate indices
        row, col = zip(*((i, le)
                         for i in range(len(self.lens))
                         for le in range(self.lens[i])))

        s, a, ret, adv, probs, mus, stds = (
            self.state[row, col], self.action[row, col], self.ret[row, col],
            self.adv[row, col], self.probs[row, col], self.mus[row, col],
            self.stds[row, col])

        # Normalize advantage. Needed ?
        # Trick used by OpenAI in their PPO impl
        # adv = (adv - adv.mean()) / (adv.std() + 1.e-8)

        shuf_ind = np.arange(s.shape[0])

        # Shuffling makes the learner unable to track in "two directions".
        # Why ?
        # np.random.shuffle(shuf_ind)

        self.clear_memory()

        return (s[shuf_ind], a[shuf_ind], ret[shuf_ind], adv[shuf_ind],
                probs[shuf_ind], mus[shuf_ind], stds[shuf_ind])

    def clear_memory(self):
        """ Reset the buffer
        """

        self.lens = np.zeros((self.n_trajectories), dtype=np.int32)
        self.ptr = 0

        # RL Buffers "filled with zeros"
        # TODO: Is that actually needed ? Can't just set self.ptr to 0 ?
        self.state = np.zeros((
            self.n_trajectories, self.max_traj_length, self.state_dim))
        self.action = np.zeros((
            self.n_trajectories, self.max_traj_length, self.action_dim))
        self.next_state = np.zeros((
            self.n_trajectories, self.max_traj_length, self.state_dim))
        self.reward = np.zeros((self.n_trajectories, self.max_traj_length))
        self.not_done = np.zeros((self.n_trajectories, self.max_traj_length))
        self.values = np.zeros((self.n_trajectories, self.max_traj_length))
        self.next_values = np.zeros(
            (self.n_trajectories, self.max_traj_length))
        self.probs = np.zeros((self.n_trajectories, self.max_traj_length))
        self.mus = np.zeros(
            (self.n_trajectories, self.max_traj_length, self.action_dim))
        self.stds = np.zeros(
            (self.n_trajectories, self.max_traj_length, self.action_dim))

        # GAE buffers
        self.ret = np.zeros((self.n_trajectories, self.max_traj_length))
        self.adv = np.zeros((self.n_trajectories, self.max_traj_length))

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
        batch_size=4096
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """ Off-policy sampling. Will sample min(batch_size, self.size)
        transitions in an unordered way. This removes the ability to do
        GAE and reward discounting after the transitions are sampled.

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

    def save_to_file(self, path):
        """ TODO for imitation learning
        """
        pass

    def load_from_file(self, path):
        """ TODO for imitation learning
        """
        pass
