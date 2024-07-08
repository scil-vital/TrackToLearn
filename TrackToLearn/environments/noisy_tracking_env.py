import numpy as np

from scipy.ndimage import map_coordinates, spline_filter
from typing import Tuple

from TrackToLearn.environments.tracking_env import TrackingEnvironment


class NoisyTrackingEnvironment(TrackingEnvironment):

    def __init__(
        self,
        dataset_file: str,
        split_id: str,
        env_dto: dict,
    ):
        """
        Parameters
        ----------
        dataset_file: str
            Path to the dataset file
        split_id: str
            Split id
        subjects: list
            List of subjects
        env_dto: dict
            Dict containing all arguments
        """

        super().__init__(dataset_file, split_id, env_dto)

        self.noise = env_dto['noise']
        self.fa_map = None
        if env_dto['fa_map']:
            self.fa_map = spline_filter(env_dto['fa_map'].data, order=3)
        self.max_action = 1.

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

        directions = actions

        if self.fa_map is not None and self.noise > 0.:
            idx = self.streamlines[self.continue_idx,
                                   self.length-1].astype(np.int32)

            # Get FA at streamline end
            fa = map_coordinates(
                self.fa_map, idx.T, prefilter=False)
            noise = ((1. - fa) * self.noise)
        else:
            noise = self.rng.normal(0., self.noise, size=directions.shape)
        directions = (
            directions + noise)
        return super().step(directions)
