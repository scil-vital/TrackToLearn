import numpy as np

from TrackToLearn.environments.interpolation import \
    nearest_neighbor_interpolation
from TrackToLearn.environments.reward import Reward
from TrackToLearn.environments.stopping_criteria import (
    is_flag_set, StoppingFlags)


class AtlasGMReward(Reward):

    """ Reward class to compute the bundle reward. Agents will be rewarded
    for reaching bundle endpoints. """

    def __init__(
        self, gm_atlas, min_nb_steps, threshold=0.5
    ):
        """
        Parameters
        ----------
        bundle_endpoint: `numpy.ndarray`
            Bundle endpoints.
        bundle_mask: `numpy.ndarray`
            Binary mask of the bundle.
        min_nb_steps: int
            Minimum number of steps for a streamline to be considered.
        threshold: float
            Threshold to consider a point as part of the bundle.
        """

        self.name = 'atlas_gm_reward'

        self.min_nb_steps = min_nb_steps

        self.gm_atlas = gm_atlas.astype(bool)

        self.threshold = threshold

    def __call__(self, streamlines, roi_idx, flags):
        """ Compute the reward for each streamline.

        Parameters
        ----------
        streamlines: `numpy.ndarray`
            Streamlines to compute the reward for.

        Returns
        -------
        rewards: `numpy.ndarray`
            Rewards for each streamline.
        """
        reward = np.zeros(len(streamlines), dtype=bool)

        coords = streamlines[:, -1, :]

        dones = is_flag_set(flags, StoppingFlags.STOPPING_MASK)

        L = streamlines.shape[1]
        if L >= self.min_nb_steps:
            reward = nearest_neighbor_interpolation(self.gm_atlas, coords)

        return reward * dones
