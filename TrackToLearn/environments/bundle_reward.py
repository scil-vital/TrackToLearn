import numpy as np

from scipy.ndimage import map_coordinates, spline_filter

from TrackToLearn.environments.reward import Reward


class BundleReward(Reward):

    """ Reward class to compute the bundle reward. Agents will be rewarded
    for reaching bundle endpoints. """

    def __init__(
        self, bundle_endpoint, bundle_mask, min_nb_steps, threshold=0.5
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

        self.name = 'bundle_reward'

        self.N = bundle_endpoint.shape[-1]
        self.min_nb_steps = min_nb_steps

        self.bundle_endpoint = [spline_filter(
            np.ascontiguousarray(
                bundle_endpoint[..., i].astype(bool), dtype=float),
            order=3)
            for i in range(self.N)]

        self.bundle_mask = [spline_filter(
            np.ascontiguousarray(
                bundle_mask[..., i].astype(bool), dtype=float), order=3)
            for i in range(self.N)]

        self.threshold = threshold

    def __call__(self, streamlines, bundle_idx, dones):
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

        if streamlines.shape[1] >= self.min_nb_steps:

            for i in range(self.N):
                b_i = bundle_idx == i
                coords = streamlines[b_i][:, -1, :].T - 0.5
                mask = map_coordinates(
                    self.bundle_endpoint[i], coords, prefilter=False
                ) > self.threshold
                reward[b_i] = mask

        return reward * dones
