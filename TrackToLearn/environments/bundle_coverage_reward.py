import numpy as np

from scipy.ndimage import map_coordinates

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

        rewards = np.zeros(len(streamlines))

        # For all bundles in the bundle mask
        for i in range(self.N):
            # Get the streamlines that are in the current bundle
            b_i = bundle_idx == i
            bundle_streamlines = streamlines[b_i]

            # Get the coordinates of the streamlines
            coordinates = bundle_streamlines[:, -1]

            # Get the coverage of the bundle
            coverage = map_coordinates(
                self.coverage[..., i], coordinates.T, order=0, mode='nearest')

            # Get the bundle mask at the coordinates
            bundle_mask = map_coordinates(
                self.bundle_mask[..., i], coordinates.T, order=0,
                mode='nearest')
            # Get the voxels that are not covered
            not_covered = coverage == 0
            in_wm = bundle_mask > 0

            # Compute the reward
            rewards[b_i] = not_covered * in_wm

            # Update the coverage of the bundle using the coordinates
            indices = coordinates.astype(int)
            self.coverage[indices.T, i] = 1

        return rewards

    def reset(self):

        pass
