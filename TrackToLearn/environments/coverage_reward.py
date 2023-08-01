import numpy as np

from fury import window, actor
from scipy.ndimage import binary_erosion

from TrackToLearn.datasets.utils import MRIDataVolume

from TrackToLearn.environments.interpolation import (
    interpolate_volume_at_coordinates)
from TrackToLearn.environments.reward import Reward


class CoverageReward(Reward):

    """ Reward streamlines based on their coverage of the tracking mask.
    """

    def __init__(
        self,
        mask: MRIDataVolume,
    ):
        self.name = 'coverage_reward'

        mask = mask.data

        wm_density = np.zeros_like(mask, dtype=int)

        while not np.all(mask == 0.):
            eroded_mask = binary_erosion(mask).astype(int)
            wm_density += eroded_mask
            mask = eroded_mask
        self.max_density = np.max(wm_density)
        self.inv_density = self.max_density - wm_density

    def __call__(
        self,
        streamlines: np.ndarray,
        dones: np.ndarray
    ):
        """
        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space

        Returns
        -------
        rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
            Array containing the reward
        """
        N, L, P = streamlines.shape
        # Get last streamlines coordinates
        density = interpolate_volume_at_coordinates(
            self.inv_density, streamlines[:, -1, :], mode='constant', order=3)
        return density / self.max_density

    def reset(self):

        pass

    def render(self, streamlines):

        scene = window.Scene()

        line_actor = actor.streamtube(
            streamlines, linewidth=1.0)
        scene.add(line_actor)

        slice_actor = actor.slicer(self.inv_density.astype(int))
        scene.add(slice_actor)

        showm = window.ShowManager(scene, reset_camera=True)
        showm.initialize()
        showm.start()