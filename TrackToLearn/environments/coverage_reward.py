import numpy as np

from fury import window, actor

from TrackToLearn.datasets.utils import MRIDataVolume
from TrackToLearn.environments.reward import Reward
from TrackToLearn.environments.utils import is_inside_mask


class CoverageReward(Reward):

    """ Reward streamlines based on their coverage of the tracking mask.
    """

    def __init__(
        self,
        mask: MRIDataVolume,
    ):
        self.name = 'coverage_reward'

        self.mask = mask.data
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

        is_in_wm = is_inside_mask(
            streamlines, self.mask, 0.5).astype(int)

        already_covered = 1 - is_inside_mask(
            streamlines, self.coverage, 0.5).astype(int)

        X, Y, Z = (streamlines[..., -1, 0],
                   streamlines[..., -1, 1],
                   streamlines[..., -1, 2])

        self.coverage[X.astype(int), Y.astype(int), Z.astype(int)] = 1.

        return is_in_wm * already_covered

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
