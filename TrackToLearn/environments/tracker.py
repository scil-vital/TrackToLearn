import numpy as np
import torch

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamline import compress_streamlines
from nibabel.streamlines import Tractogram
from typing import Tuple

from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.environments.utils import StoppingFlags, is_flag_set
from TrackToLearn.utils.utils import normalize_vectors


class Tracker(BaseEnv):
    """
    Tracking environment.
    TODO: Clean up "_private functions" and public functions
    """

    def _is_stopping(
        self,
        streamlines: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Check which streamlines should stop or not according to the
        predefined stopping criteria

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamlines that will be checked

        Returns
        -------
        continue_idx : `numpy.ndarray`
            Indices of the streamlines that should continue
        stopping_idx : `numpy.ndarray`
            Indices of the streamlines that should stop
        stopping_flags : `numpy.ndarray`
            `StoppingFlags` that triggered stopping for each stopping
            streamline
        """
        continue_idx, stopping_idx, stopping_flags = \
            self._filter_stopping_streamlines(
                streamlines, self.stopping_criteria)
        return continue_idx, stopping_idx, stopping_flags

    def _keep(
        self,
        idx: np.ndarray,
        state: np.ndarray,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> np.ndarray:
        """ Keep only streamlines corresponding to the given indices, and remove
        all others. The model states will be updated accordingly.

        Parameters
        ----------
        idx : `np.ndarray`
            Indices of the streamlines to keep
        streamlines: Tractogram
            Tractograms to filter
        state: np.ndarray
            Whole batch to filter

        Returns:
        --------
        streamlines: Tractogram
            Tractogram to filter
        state: np.ndarray
            Batched state to filter
        """

        self.streamlines = self.streamlines[idx]
        state = state[idx]
        self.dones = self.dones[idx]
        if hidden is not None:
            h, c = hidden
            h = h[:, idx, :]
            c = c[:, idx, :]
            hidden = (h, c)

        return state, hidden

    def nreset(self, batch_size: int) -> np.ndarray:
        """ Initialize tracking seeds and streamlines
        Parameters
        ----------
        batch_size: int
            How many seeds to sample

        Returns
        -------
        state: numpy.ndarray
            Initial state for RL model
        """
        seeds = np.random.randint(0, len(self.seeds),
                                  size=min(len(self.seeds), batch_size))

        self.streamlines = np.zeros(
            (seeds.shape[0], self.max_nb_steps + 1, 3), dtype=np.float32)
        self.streamlines[:, 0, :] = self.seeds[seeds]

        self.done_streamlines = self.streamlines.copy()
        self.lengths = np.ones(seeds.shape[0], dtype=np.int)
        self.flags = np.zeros(seeds.shape[0], dtype=np.uint8)
        self.done_idx = 0

        self.length = 1

        # Initialize streamline seeds
        self.starting_points = self.streamlines[:, 0, :]

        # Initialize rewards and done flags
        self.dones = np.full(self.streamlines.shape[0], False)

        # Setup input signal
        return self._format_state(self.streamlines[:, :self.length])

    def reset(self, start: int, end: int) -> np.ndarray:
        """ Initialize tracking seeds and streamlines
        Parameters
        ----------
        start: int
            Starting index of seed to add to batch
        end: int
            Ending index of seeds to add to batch

        Returns
        -------
        state: numpy.ndarray
            Initial state for RL model
        """
        # Initialize seeds as streamlines
        first_points = self.seeds[start:end]
        self.streamlines = np.zeros(
            (first_points.shape[0], self.max_nb_steps + 1, 3),
            dtype=np.float32)
        self.streamlines[:, 0, :] = first_points

        self.done_streamlines = self.streamlines.copy()
        self.lengths = np.ones(first_points.shape[0], dtype=np.int)
        self.flags = np.zeros(first_points.shape[0], dtype=np.uint8)
        self.done_idx = 0

        self.length = 1

        # Initialize streamline seeds
        self.starting_points = self.streamlines[:, 0, :]

        # Initialize rewards and done flags
        self.dones = np.full(self.streamlines.shape[0], False)

        # Setup input signal
        return self._format_state(self.streamlines[:, :self.length])

    def step(
        self,
        directions: np.ndarray,
    ) -> Tuple[np.ndarray, list, bool, dict]:
        """
        Apply actions and grow streamlines for one step forward
        Calculate rewards and if the tracking is done, and compute new
        hidden states

        Parameters
        ----------
        directions: np.ndarray
            Actions applied to the state

        Returns
        -------
        state: np.ndarray
            New state
        reward: list
            Reward for the last step of the streamline
        done: bool
            Whether the episode is done
        info: dict
        """
        # Scale directions to step size
        directions = normalize_vectors(directions) * self.step_size_vox

        # Grow streamlines one step forward
        self.streamlines[:, self.length, :] = \
            self.streamlines[:, self.length-1, :] + directions
        self.length += 1

        # Get stopping and keeping indexes
        self.continue_idx, self.stopping_idx, self.stopping_flags = \
            self._is_stopping(self.streamlines[:, :self.length])

        # Compute streamlines that are done for real
        self.dones[self.stopping_idx] = 1

        reward = np.zeros(self.streamlines.shape[0])
        # Compute reward if wanted. At test time, no need
        # to compute it and slow down the tracking process
        if self.compute_reward:
            reward = self.reward_function(
                self.streamlines[:, :self.length, :], self.dones)

        return (
            self._format_state(self.streamlines[:, :self.length]),
            reward, self.dones, {})

    def harvest(
        self,
        states: np.ndarray,
        hidden,
    ) -> Tuple[StatefulTractogram, np.ndarray]:
        """Internally keep only the streamlines and corresponding env. states
        that haven't stopped yet, and return the streamlines that triggered a
        stopping flag.

        Parameters
        ----------
        states: torch.Tensor
            Environment states to be "pruned"

        Returns
        -------
        tractogram : nib.streamlines.Tractogram
            Tractogram containing the streamlines that stopped tracking,
            along with the stopping_flags information and seeds in
            `tractogram.data_per_streamline`
        states: np.ndarray of size [n_streamlines, input_size]
            Input size for all continuing last streamline positions and
            neighbors + input addons
        stopping_idx: np.ndarray
            Indexes of stopping trajectories. Returned in case an RL
            algorithm would need 'em
        """
        all_id = np.arange(len(self.streamlines))
        done_idx = np.setdiff1d(
            all_id, self.continue_idx, assume_unique=True).astype(np.int)
        N_dones = len(done_idx)

        self.done_streamlines[
            self.done_idx: self.done_idx + N_dones, :self.length, :3] = \
            self.streamlines[done_idx, :self.length, :]
        self.lengths[
            self.done_idx: self.done_idx + N_dones] = self.length
        self.flags[self.done_idx: self.done_idx + N_dones] = \
            self.stopping_flags
        self.done_idx += N_dones

        # Keep only streamlines that should continue
        states, hidden = self._keep(
            self.continue_idx,
            states,
            hidden)

        return states, hidden, self.continue_idx

    def get_streamlines(self, compress=False) -> StatefulTractogram:

        tractogram = Tractogram()
        # Harvest stopped streamlines and associated data
        stopped_seeds = self.starting_points
        stopped_streamlines = [self.done_streamlines[i, :self.lengths[i], :]
                               for i in range(len(self.done_streamlines))]
        # Drop last point if it triggered a flag we don't want
        flags = is_flag_set(
            self.flags, StoppingFlags.STOPPING_CURVATURE)
        streamlines = [
            s[:-1] if f else s for f, s in zip(flags, stopped_streamlines)]

        if compress:
            streamlines = compress_streamlines(streamlines, 0.1)

        # Harvested tractogram
        tractogram = Tractogram(
            streamlines=streamlines,
            data_per_streamline={"stopping_flags": self.flags,
                                 "seeds": stopped_seeds
                                 },
            affine_to_rasmm=self.affine_vox2rasmm)

        return tractogram


class Retracker(Tracker):
    """ Pre-initialized environment
    Tracking will start from the end of streamlines for two reasons:
        - For computational purposes, it's easier if all streamlines have
          the same length and are harvested as they end
        - Tracking back the streamline and computing the alignment allows some
          sort of "self-supervised" learning for tracking backwards
    """

    def _is_stopping(
        self,
        streamlines: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Check which streamlines should stop or not according to the
        predefined stopping criteria

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamlines that will be checked

        Returns
        -------
        continue_idx : `numpy.ndarray`
            Indices of the streamlines that should continue
        stopping_idx : `numpy.ndarray`
            Indices of the streamlines that should stop
        stopping_flags : `numpy.ndarray`
            `StoppingFlags` that triggered stopping for each stopping
            streamline
        """
        continue_idx, stopping_idx, stopping_flags = \
            super()._is_stopping(streamlines)

        # Indices for streamlines that are still being initialized
        is_still_initializing = self.n_init_steps > streamlines.shape[1]

        # Streamlines that haven't finished initializing should keep going
        continue_idx = np.concatenate(
            [continue_idx, [idx for
                            idx in stopping_idx if
                            is_still_initializing[idx]]])
        continue_idx = continue_idx.astype(int)

        # Streamlines that haven't finished initializing should not stop
        is_really_stopping = np.logical_not(
            is_still_initializing[stopping_idx])
        stopping_idx = stopping_idx[is_really_stopping]
        stopping_flags = stopping_flags[is_really_stopping]

        return continue_idx, stopping_idx, stopping_flags

    def _keep(
        self,
        idx: np.ndarray,
        states: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Keep only streamlines corresponding to the given indices, and remove
        all others. The model states will be updated accordingly.

        Parameters
        ----------
        idx : `np.ndarray`
            Indices of the streamlines to keep
        states: np.ndarray
            Input state batch items to keep
        """

        self.seeding_streamlines = [self.seeding_streamlines[i] for i in idx]
        self.n_init_steps = self.n_init_steps[idx]

        return super()._keep(
            idx, states)

    def reset(self, streamlines: np.ndarray) -> np.ndarray:
        """ Initialize tracking seeds and streamlines
        Parameters
        ----------
        seeding_streamlines: np.ndarray
            Half-streamlines to initialize environment

        Returns
        -------
        state: numpy.ndarray
            Initial state for RL model
        """

        # Half-streamlines
        seeding_streamlines = [s[::-1] for s in streamlines]
        self.seeding_streamlines = seeding_streamlines

        self.starting_points = np.array([s[-1] for s in seeding_streamlines])

        # Number if initialization steps for each streamline
        self.n_init_steps = np.asarray(list(map(len, seeding_streamlines)))

        # Get the first point of each seed as the start of the new streamlines
        self.streamlines = np.zeros(
            (self.starting_points.shape[0], self.max_nb_steps, 3),
            dtype=np.float32)

        self.lengths = np.ones(self.starting_points.shape[0], dtype=np.int)
        self.flags = np.zeros(self.starting_points.shape[0], dtype=np.uint8)
        self.done_idx = 0

        for i, s in enumerate(seeding_streamlines):
            le = len(s)
            self.streamlines[i, :le, :] = s

        self.done_streamlines = self.streamlines.copy()
        # Done flags for tracking backwards
        self.dones = np.full(len(self.streamlines), False)

        self.length = 1

        # Signal
        return self._format_state(self.streamlines[:, :self.length])

    def step(
        self,
        directions: np.ndarray,
    ) -> Tuple[np.ndarray, list, bool, dict]:
        """
        Apply actions and grow streamlines for one step forward
        Calculate rewards and if the tracking is done. While tracking
        has not surpassed half-streamlines, replace the tracking step
        taken with the actual streamline segment

        Parameters
        ----------
        directions: np.ndarray
            Actions applied to the state

        Returns
        -------
        state: np.ndarray
            New state
        reward: list
            Reward for the last step of the streamline
        done: bool
            Whether the episode is done
        info: dict
        """
        # Scale directions to step size
        normalized_directions = directions / np.sqrt(np.sum(
            directions ** 2,
            axis=-1,
            keepdims=True))
        directions = normalized_directions * self.step_size_vox

        # Grow streamlines one step forward
        self.streamlines[:, self.length, :] = \
            self.streamlines[:, self.length-1, :] + directions
        self.length += 1

        # Do not update streamlines that are still initializing
        is_still_initializing = self.n_init_steps > self.length + 1
        if np.any(is_still_initializing):
            # Replace the last point of the predicted streamlines with
            # the seeding streamlines at the same position
            is_still_initializing_idx = \
                np.arange(len(self.n_init_steps))[is_still_initializing]

            for i in is_still_initializing_idx:
                self.streamlines[i][self.length-1] = \
                    self.seeding_streamlines[i][self.length-1]

        # Get indices for streamlines that are done
        self.continue_idx, self.stopping_idx, self.stopping_flags = \
            self._is_stopping(self.streamlines[:, :self.length])

        # Compute streamlines that are done for real
        self.dones[self.stopping_idx] = 1

        reward = np.zeros(self.streamlines.shape[0])
        if self.compute_reward:
            # Reward streamline step
            reward = self.reward_function(
                self.streamlines[:, :self.length, :], self.dones)

        return (
            self._format_state(self.streamlines[:, :self.length]),
            reward, self.dones, {})


class BackwardTracker(Tracker):
    """ Pre-initialized environment
    Tracking will start at the seed. To add directionality to the signal, the
    beginning of the half-streamlines is used. When harvesting, both parts of
    the streamlines are concatenated. """

    def reset(self, streamlines: np.ndarray) -> np.ndarray:
        """ Initialize tracking seeds and streamlines
        Parameters
        ----------
        streamlines : list
            Half-streamlines to initialize environment

        Returns
        -------
        state: numpy.ndarray
            Initial state for RL model
        """

        # Half-streamlines
        self.seeding_streamlines = [s[:] for s in streamlines]
        N = len(streamlines)

        # Jagged arrays ugh
        # This is dirty, clean up asap
        self.half_lengths = np.asarray(
            [len(s) for s in self.seeding_streamlines])
        max_half_len = max(self.half_lengths)
        half_streamlines = np.zeros(
            (N, max_half_len, 3), dtype=np.float32)

        for i, s in enumerate(self.seeding_streamlines):
            le = self.half_lengths[i]
            half_streamlines[i, :le, :] = s

        self.starting_points = np.asarray([s[0] for s in streamlines])

        # Initialize seeds as streamlines
        self.streamlines = np.concatenate((np.zeros(
            (N, self.max_nb_steps + 1, 3),
            dtype=np.float32), half_streamlines), axis=1)

        self.streamlines = np.flip(self.streamlines, axis=1)

        self.done_streamlines = self.streamlines.copy()
        self.lengths = np.ones(N, dtype=np.int) * max_half_len
        self.flags = np.zeros(N, dtype=np.uint8)
        self.done_idx = 0

        # Done flags for tracking backwards
        self.dones = np.full(len(self.streamlines), False)
        self.max_half_len = max_half_len
        self.length = max_half_len

        # Signal
        return self._format_state(self.streamlines[:, :self.length])

    def harvest(
        self,
        states: np.ndarray,
        hidden,
        compress=False,
    ) -> Tuple[StatefulTractogram, np.ndarray]:
        """Internally keep only the streamlines and corresponding env. states
        that haven't stopped yet, and return the streamlines that triggered a
        stopping flag.

        Parameters
        ----------
        states: torch.Tensor
            Environment states to be "pruned"

        Returns
        -------
        tractogram : nib.streamlines.Tractogram
            Tractogram containing the streamlines that stopped tracking,
            along with the stopping_flags information and seeds in
            `tractogram.data_per_streamline`
        states: np.ndarray of size [n_streamlines, input_size]
            Input size for all continuing last streamline positions and
            neighbors + input addons
        stopping_idx: np.ndarray
            Indexes of stopping trajectories. Returned in case an RL
            algorithm would need 'em
        """
        all_id = np.arange(len(self.streamlines))
        done_idx = np.setdiff1d(
            all_id, self.continue_idx, assume_unique=True)

        for i, d in enumerate(done_idx):
            length = self.length - (self.max_half_len - self.half_lengths[d])
            self.done_streamlines[self.done_idx + i, :length, :] = \
                self.streamlines[
                    d,
                    self.max_half_len - self.half_lengths[d]:self.length, :]
            self.lengths[self.done_idx + i] = length
            self.flags[self.done_idx + i] = self.stopping_flags[i]

        self.done_idx += len(done_idx)
        # Keep only streamlines that should continue
        states, hidden = self._keep(
            self.continue_idx,
            states)

        return states, hidden, self.continue_idx
