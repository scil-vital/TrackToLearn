import numpy as np

from fury import window, actor
from scipy.ndimage import binary_erosion, map_coordinates, spline_filter

from TrackToLearn.datasets.utils import MRIDataVolume

from TrackToLearn.environments.reward import Reward


class CoverageReward(Reward):

    """ Reward streamlines based on their coverage of the tracking mask.

    **IMPORTANT**: This has not been published but it works reasonably well.
    If you want to include this in your publication, please contact me
    beforehand.
    """

    def __init__(
        self,
        mask: MRIDataVolume,
    ):
        self.name = 'coverage_reward'

        self.mask = mask.data.astype(int)

        erosion = binary_erosion(self.mask)

        self.coverage = spline_filter(
            np.ascontiguousarray((self.mask - erosion).astype(int)), order=3)

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

        coords = streamlines[:, -1, :] - 0.5

        coverage = map_coordinates(
            self.coverage, coords.T, mode='constant', prefilter=False)
        return coverage

    def reset(self):

        pass

    def render(self, streamlines):

        scene = window.Scene()

        line_actor = actor.streamtube(
            streamlines, linewidth=1.0)
        scene.add(line_actor)

        slice_actor = actor.slicer(self.coverage.astype(int))
        scene.add(slice_actor)

        showm = window.ShowManager(scene, reset_camera=True)
        showm.initialize()
        showm.start()
