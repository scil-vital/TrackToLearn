import numpy as np

from fury import window, actor

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

        self.mask = 1. - mask.data
        self.coverage = np.zeros_like(self.mask)

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
        borders = interpolate_volume_at_coordinates(
            self.mask, streamlines[:, -1, :], mode='constant', order=3)

        return borders

    def reset(self):

        self.coverage = np.zeros_like(self.mask)

    def render(self, streamlines):

        scene = window.Scene()

        line_actor = actor.streamtube(
            streamlines, linewidth=1.0)
        scene.add(line_actor)

        slice_actor = actor.slicer(self.coverage)
        scene.add(slice_actor)

        showm = window.ShowManager(scene, reset_camera=True)
        showm.initialize()
        showm.start()
