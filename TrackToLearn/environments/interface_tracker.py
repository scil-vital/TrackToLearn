import numpy as np

from typing import Tuple

from TrackToLearn.environments.tracker import Tracker
from TrackToLearn.environments.noisy_tracker import NoisyTracker
from TrackToLearn.utils.utils import normalize_vectors


class InterfaceTracker(Tracker):

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

        # If the streamline goes out the tracking mask at the first
        # step, flip it
        if self.length == 1:
            # Scale directions to step size
            directions = normalize_vectors(directions) * self.step_size_vox

            # Grow streamlines one step forward
            streamlines = self.streamlines.copy()
            streamlines[:, self.length, :] = \
                self.streamlines[:, self.length-1, :] + directions

            # Get stopping and keeping indexes
            continue_idx, stopping_idx, stopping_flags = \
                self._is_stopping(
                    streamlines[:, :self.length + 1])

            # Flip stopping trajectories
            directions[stopping_idx] *= -1

        return super().step(directions)


class InterfaceNoisyTracker(NoisyTracker):

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

        # If the streamline goes out the tracking mask at the first
        # step, flip it
        if self.length == 1:
            # Scale directions to step size
            directions = normalize_vectors(directions) * self.step_size_vox

            # Grow streamlines one step forward
            streamlines = self.streamlines.copy()
            streamlines[:, self.length, :] = \
                self.streamlines[:, self.length-1, :] + \
                directions

            # Get stopping and keeping indexes
            continue_idx, stopping_idx, stopping_flags = \
                self._is_stopping(
                    streamlines[:, :self.length + 1])

            # Flip stopping trajectories
            directions[stopping_idx] *= -1

        return super().step(directions)
