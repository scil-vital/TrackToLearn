import numpy as np
import torch

from typing import Tuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.g = np.random.Generator(np.random.PCG64())

        self.device = device
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        # Buffers "filled with zeros"
        self.state = torch.zeros(
            (self.max_size, state_dim), dtype=torch.float32).pin_memory()
        self.action = torch.zeros(
            (self.max_size, action_dim), dtype=torch.float32).pin_memory()
        self.next_state = torch.zeros(
            (self.max_size, state_dim), dtype=torch.float32).pin_memory()
        self.reward = torch.zeros(
            (self.max_size, 1), dtype=torch.float32).pin_memory()
        self.not_done = torch.zeros(
            (self.max_size, 1), dtype=torch.float32).pin_memory()

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

    def __len__(self):
        return self.size

    def sample(
        self,
        batch_size=4096
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """ Off-policy sampling. Will sample min(batch_size, self.size)
        transitions in an unordered way.

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
        ind = self.g.choice(
            self.size, min(self.size, batch_size), replace=False)
        # ind = torch.randperm(self.size, dtype=torch.long)[
        #     :min(self.size, batch_size)]
        ind = torch.as_tensor(ind)

        s = self.state.index_select(0, ind).pin_memory()
        a = self.action.index_select(0, ind).pin_memory()
        ns = self.next_state.index_select(0, ind).pin_memory()
        r = self.reward.index_select(0, ind).squeeze(-1).pin_memory()
        d = self.not_done.index_select(0, ind).to(
            dtype=torch.float32).squeeze(-1).pin_memory()

        # Return tensors on the same device as the buffer in pinned memory
        return (s.to(device=self.device, non_blocking=True),
                a.to(device=self.device, non_blocking=True),
                ns.to(device=self.device, non_blocking=True),
                r.to(device=self.device, non_blocking=True),
                d.to(device=self.device, non_blocking=True))

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
