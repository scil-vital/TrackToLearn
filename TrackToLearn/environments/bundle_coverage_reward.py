import numpy as np

from TrackToLearn.environments.interpolation import (
    nearest_neighbor_interpolation)
from TrackToLearn.environments.reward import Reward


class BundleCoverageReward(Reward):

    """ Reward streamlines based on their coverage of the bundle mask.
    Streamlines are rewarded for entering a voxel of the bundle mask where
    the coverage is zero. The coverage is updated after each step.
    """

    def __init__(
        self,
        bundle_mask: np.ndarray,
    ):
        self.name = 'bundle_coverage_reward'
        self.bundle_mask = bundle_mask.astype(bool)
        self.N = bundle_mask.shape[-1]
        self.coverage = np.zeros_like(bundle_mask)

    def __call__(
        self,
        streamlines: np.ndarray,
        bundle_idx: np.ndarray,
        dones: np.ndarray
    ):
        """ Compute the reward for each streamline. """

        bundle_idx = bundle_idx.astype(int)
        all_idx = np.arange(bundle_idx.shape[0])

        rewards = np.zeros(len(streamlines))

        # Get the coordinates of the streamlines
        coordinates = (streamlines[:, -1] + 0.5)
        upper = (np.asarray(self.coverage.shape[:3]) - 1)
        coordinates = np.clip(coordinates.astype(int), 0, upper).astype(int)
        indices = np.concatenate((coordinates,
                                 bundle_idx[..., None]), axis=-1).T

        # Get the coverage of the bundle
        coverage = nearest_neighbor_interpolation(
            self.coverage, coordinates)
        bundle_mask = nearest_neighbor_interpolation(
            self.bundle_mask, coordinates)

        # Get the voxels that are not covered
        not_covered = (coverage[all_idx, bundle_idx] == 0).squeeze()
        in_wm = (bundle_mask[all_idx, bundle_idx] > 0).squeeze()

        # Compute the reward
        rewards = not_covered * in_wm
        self.coverage[tuple(indices)] = 1

        return rewards

    def reset(self):
        self.coverage = np.zeros_like(self.bundle_mask)
