import numpy as np

from typing import Tuple

from TrackToLearn.environments.tracking_env import TrackingEnvironment
from TrackToLearn.environments.noisy_tracker import NoisyTrackingEnvironment


class InterfaceTrackingEnvironment(TrackingEnvironment):

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, list, bool, dict]:
        """
        Apply actions and grow streamlines for one step forward
        Calculate rewards and if the tracking is done, and compute new
        hidden states

        Parameters
        ----------
        actions: np.ndarray
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

        directions = self._format_actions(actions)

        # If the streamline goes out the tracking mask at the first
        # step, flip it
        if self.length == 1:
            # Grow streamlines one step forward
            streamlines = self.streamlines[self.continue_idx].copy()
            streamlines[:, self.length, :] = \
                self.streamlines[self.continue_idx,
                                 self.length-1, :] + directions

            # Get stopping and keeping indexes
            stopping, flags = \
                self._is_stopping(
                    streamlines[:, :self.length + 1])

            # Flip stopping trajectories
            directions[stopping] *= -1

        return super().step(directions)


class InterfaceNoisyTrackingEnvironment(NoisyTrackingEnvironment):

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, list, bool, dict]:
        """
        Apply actions and grow streamlines for one step forward
        Calculate rewards and if the tracking is done, and compute new
        hidden states

        Parameters
        ----------
        actions: np.ndarray
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

        directions = self._format_actions(actions)

        # If the streamline goes out the tracking mask at the first
        # step, flip it
        if self.length == 1:
            # Grow streamlines one step forward
            streamlines = self.streamlines[self.continue_idx].copy()
            streamlines[:, self.length, :] = \
                self.streamlines[self.continue_idx,
                                 self.length-1, :] + directions

            # Get stopping and keeping indexes
            stopping, flags = \
                self._is_stopping(
                    streamlines[:, :self.length + 1])

            # Flip stopping trajectories
            directions[stopping] *= -1

        return super().step(directions)
