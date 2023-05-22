import numpy as np

from typing import Tuple

from dipy.io.stateful_tractogram import StatefulTractogram
from nibabel.streamlines import Tractogram

from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.environments.stopping_criteria import (
    is_flag_set,
    StoppingFlags)
from TrackToLearn.utils.utils import normalize_vectors


class TrackingEnvironment(BaseEnv):
    """ Tracking environment.
    TODO: Clean up "_private functions" and public functions. Some could
    go into BaseEnv.
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
        stopping: `numpy.ndarray`
            Mask of stopping streamlines.
        stopping_flags : `numpy.ndarray`
            `StoppingFlags` that triggered stopping for each stopping
            streamline.
        """
        stopping, flags = \
            self._filter_stopping_streamlines(
                streamlines, self.stopping_criteria)
        return stopping, flags

    def _keep(
        self,
        idx: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        """ Keep only states that correspond to continuing streamlines.

        Parameters
        ----------
        idx : `np.ndarray`
            Indices of the streamlines/states to keep
        state: np.ndarray
            Batch of states.

        Returns:
        --------
        state: np.ndarray
            Continuing states.
        """
        state = state[idx]

        return state

    def nreset(self, n_seeds: int) -> np.ndarray:
        """ Initialize tracking seeds and streamlines. Will
        chose N random seeds among all seeds.

        Parameters
        ----------
        n_seeds: int
            How many seeds to sample

        Returns
        -------
        state: numpy.ndarray
            Initial state for RL model
        """

        # Heuristic to avoid duplicating seeds if fewer seeds than actors.
        replace = n_seeds > len(self.seeds)
        seeds = np.random.choice(
            np.arange(len(self.seeds)), size=n_seeds, replace=replace)
        self.initial_points = self.seeds[seeds]

        self.streamlines = np.zeros(
            (n_seeds, self.max_nb_steps + 1, 3), dtype=np.float32)
        self.streamlines[:, 0, :] = self.initial_points

        self.flags = np.zeros(n_seeds, dtype=int)

        self.lengths = np.ones(n_seeds, dtype=np.int32)

        self.length = 1

        # Initialize rewards and done flags
        self.dones = np.full(n_seeds, False)
        self.continue_idx = np.arange(n_seeds)

        # Setup input signal
        return self._format_state(
            self.streamlines[self.continue_idx, :self.length])

    def reset(self, start: int, end: int) -> np.ndarray:
        """ Initialize tracking seeds and streamlines. Will select
        a given batch of seeds.

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
        self.initial_points = self.seeds[start:end]
        N = self.initial_points.shape[0]

        self.streamlines = np.zeros(
            (N, self.max_nb_steps + 1, 3),
            dtype=np.float32)
        self.streamlines[:, 0, :] = self.initial_points
        self.flags = np.zeros(N, dtype=int)

        self.lengths = np.ones(N, dtype=np.int32)
        self.length = 1

        # Initialize rewards and done flags
        self.dones = np.full(N, False)
        self.continue_idx = np.arange(N)

        # Setup input signal
        return self._format_state(
            self.streamlines[self.continue_idx, :self.length])

    def step(
        self,
        directions: np.ndarray,
    ) -> Tuple[np.ndarray, list, bool, dict]:
        """
        Apply actions, rescale actions to step size and grow streamlines
        for one step forward. Calculate rewards and stop streamlines.

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
        directions = normalize_vectors(directions) * self.step_size

        # Grow streamlines one step forward
        self.streamlines[self.continue_idx, self.length, :] = \
            self.streamlines[self.continue_idx, self.length-1, :] + directions
        self.length += 1

        # Get stopping and keeping indexes
        stopping, new_flags = \
            self._is_stopping(
                self.streamlines[self.continue_idx, :self.length])

        self.new_continue_idx, self.stopping_idx = \
            (self.continue_idx[~stopping],
             self.continue_idx[stopping])

        mask_continue = np.in1d(
            self.continue_idx, self.new_continue_idx, assume_unique=True)
        diff_stopping_idx = np.arange(
            len(self.continue_idx))[~mask_continue]

        self.flags[
            self.stopping_idx] = new_flags[diff_stopping_idx]

        self.dones[self.stopping_idx] = 1

        reward = np.zeros(self.streamlines.shape[0])
        # Compute reward if wanted. At valid time, no need
        # to compute it and slow down the tracking process
        if self.compute_reward:
            reward = self.reward_function(
                self.streamlines[self.continue_idx, :self.length],
                self.dones[self.continue_idx])

        return (
            self._format_state(
                self.streamlines[self.continue_idx, :self.length]),
            reward, self.dones[self.continue_idx],
            {'continue_idx': self.continue_idx})

    def harvest(
        self,
        states: np.ndarray,
    ) -> Tuple[StatefulTractogram, np.ndarray]:
        """Internally keep only the streamlines and corresponding env. states
        that haven't stopped yet, and return the states that continue.

        Parameters
        ----------
        states: torch.Tensor
            States before "pruning" or "harvesting".

        Returns
        -------
        states: np.ndarray of size [n_streamlines, input_size]
            States corresponding to continuing streamlines.
        continue_idx: np.ndarray
            Indexes of trajectories that did not stop.
        """

        # Register the length of the streamlines that have stopped.
        self.lengths[self.stopping_idx] = self.length

        mask_continue = np.in1d(
            self.continue_idx, self.new_continue_idx, assume_unique=True)
        diff_continue_idx = np.arange(
            len(self.continue_idx))[mask_continue]
        self.continue_idx = self.new_continue_idx

        # Keep only streamlines that should continue
        states = self._keep(
            diff_continue_idx,
            states)

        return states, diff_continue_idx

    def get_streamlines(self) -> StatefulTractogram:
        """ Obtain tracked streamlines fromm the environment.
        The last point will be removed if it raised a curvature stopping
        criteria (i.e. the angle was too high). Otherwise, other last points
        are kept.

        TODO: remove them also ?

        Returns
        -------
        tractogram: Tractogram
            Tracked streamlines.

        """

        tractogram = Tractogram()
        # Harvest stopped streamlines and associated data
        # stopped_seeds = self.first_points[self.stopping_idx]
        # Exclude last point as it triggered a stopping criteria.
        stopped_streamlines = [self.streamlines[i, :self.lengths[i], :]
                               for i in range(len(self.streamlines))]

        flags = is_flag_set(
            self.flags, StoppingFlags.STOPPING_CURVATURE)
        stopped_streamlines = [
            s[:-1] if f else s for f, s in zip(flags, stopped_streamlines)]

        stopped_seeds = self.initial_points

        # Harvested tractogram
        tractogram = Tractogram(
            streamlines=stopped_streamlines,
            data_per_streamline={"seeds": stopped_seeds,
                                 },
            affine_to_rasmm=self.affine_vox2rasmm)

        return tractogram
