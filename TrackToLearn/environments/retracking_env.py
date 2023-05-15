import numpy as np

from typing import Tuple

from TrackToLearn.environments.tracking_env import TrackingEnvironment
from TrackToLearn.utils.utils import normalize_vectors


class RetrackingEnvironment(TrackingEnvironment):
    """ Pre-initialized environment
    Tracking will start from the end of streamlines for two reasons:
        - For computational purposes, it's easier if all streamlines have
          the same length and are harvested as they end
        - Tracking back the streamline and computing the alignment allows some
          sort of "self-supervised" learning for tracking backwards
    """

    def _is_stopping(
        self,
        streamlines: np.ndarray,
        is_still_initializing: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Check which streamlines should stop or not according to the
        predefined stopping criteria. An additional check is performed
        to prevent stopping if the retracking process is not over.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamlines that will be checked
        is_still_initializing: `numpy.ndarray` of shape (n_streamlines)
            Mask that indicates which streamlines are still being
            retracked.

        Returns
        -------
        stopping: `numpy.ndarray`
            Mask of stopping streamlines.
        stopping_flags : `numpy.ndarray`
            `StoppingFlags` that triggered stopping for each stopping
            streamline.
        """
        stopping, flags = super()._is_stopping(streamlines)

        # Streamlines that haven't finished initializing should keep going
        stopping[is_still_initializing[self.continue_idx]] = False
        flags[is_still_initializing[self.continue_idx]] = 0

        return stopping, flags

    def reset(self, half_streamlines: np.ndarray) -> np.ndarray:
        """ Initialize tracking from half-streamlines.

        Parameters
        ----------
        half_streamlines: np.ndarray
            Half-streamlines to initialize environment

        Returns
        -------
        state: numpy.ndarray
            Initial state for RL model
        """

        # Half-streamlines
        self.initial_points = np.array([s[0] for s in half_streamlines])

        # Number if initialization steps for each streamline
        self.n_init_steps = np.asarray(list(map(len, half_streamlines)))

        N = len(self.n_init_steps)

        # Get the first point of each seed as the start of the new streamlines
        self.streamlines = np.zeros(
            (N, self.max_nb_steps, 3),
            dtype=np.float32)

        for i, (s, l) in enumerate(zip(half_streamlines, self.n_init_steps)):
            self.streamlines[i, :l, :] = s[::-1]

        self.seeding_streamlines = self.streamlines.copy()

        self.lengths = np.ones(N, dtype=np.int32)
        self.length = 1

        # Done flags for tracking backwards
        self.flags = np.zeros(N, dtype=int)
        self.dones = np.full(N, False)
        self.continue_idx = np.arange(N)

        # Signal
        return self._format_state(
            self.streamlines[self.continue_idx, :self.length])

    def step(
        self,
        directions: np.ndarray,
    ) -> Tuple[np.ndarray, list, bool, dict]:
        """
        Apply actions and grow streamlines for one step forward
        Calculate rewards and if the tracking is done. While tracking
        has not surpassed half-streamlines, replace the tracking step
        taken with the actual streamline segment.

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
        self.streamlines[self.continue_idx, self.length,
                         :] = self.streamlines[
                             self.continue_idx, self.length-1, :] + directions
        self.length += 1

        # Check which streamline are still being retracked
        is_still_initializing = self.n_init_steps > self.length + 1

        # Get stopping and keeping indexes
        # self._is_stopping is overridden to take into account retracking
        stopping, new_flags = self._is_stopping(
            self.streamlines[self.continue_idx, :self.length],
            is_still_initializing)

        self.new_continue_idx, self.stopping_idx = (
            self.continue_idx[~stopping],
            self.continue_idx[stopping])

        mask_continue = np.in1d(
            self.continue_idx, self.new_continue_idx, assume_unique=True)
        diff_stopping_idx = np.arange(
            len(self.continue_idx))[~mask_continue]

        # Set "done" flags for RL
        self.dones[self.stopping_idx] = 1

        # Store stopping flags
        self.flags[
            self.stopping_idx] = new_flags[diff_stopping_idx]

        # Compute reward
        reward = np.zeros(self.streamlines.shape[0])
        if self.compute_reward:
            # Reward streamline step
            reward = self.reward_function(
                self.streamlines[self.continue_idx, :self.length, :],
                self.dones)

        # If a streamline is still being retracked
        if np.any(is_still_initializing):
            # Replace the last point of the predicted streamlines with
            # the seeding streamlines at the same position

            self.streamlines[is_still_initializing, self.length - 1] = \
                self.seeding_streamlines[is_still_initializing,
                                         self.length - 1]

        # Return relevant infos
        return (
            self._format_state(
                self.streamlines[self.continue_idx, :self.length]),
            reward, self.dones[self.continue_idx],
            {'continue_idx': self.continue_idx})
