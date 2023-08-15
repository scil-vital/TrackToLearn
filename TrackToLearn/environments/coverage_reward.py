import numpy as np
import torch

from fury import window, actor
from scipy.ndimage import binary_erosion

from TrackToLearn.environments.interpolation import (
    torch_trilinear_interpolation)
from TrackToLearn.environments.reward import Reward


class CoverageReward(Reward):

    """ Reward streamlines based on their coverage of the tracking mask.
    """

    def __init__(
        self,
        mask: torch.Tensor,
    ):
        self.name = 'coverage_reward'

        wm_density = np.zeros_like(mask.cpu().numpy(), dtype=int)
        np_mask = mask.cpu().numpy()

        while not np.all(np_mask == 0.):
            # Binary erosion does not work with tensors
            # TODO: Fix
            eroded_mask = binary_erosion(np_mask).astype(int)
            wm_density += eroded_mask
            np_mask = eroded_mask
        self.max_density = np.max(wm_density)
        self.inv_density = self.max_density - torch.as_tensor(
            wm_density, device=mask.device)

    def __call__(
        self,
        streamlines: torch.Tensor,
        dones: torch.Tensor
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
        density = torch_trilinear_interpolation(
            self.inv_density, streamlines[:, -1, :])
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
