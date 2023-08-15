# import numpy as np
import torch

from TrackToLearn.environments.interpolation import (
    torch_nearest_interpolation, torch_trilinear_interpolation)
from TrackToLearn.datasets.utils import MRIDataVolume
from TrackToLearn.environments.reward import Reward
from TrackToLearn.utils.utils import normalize_vectors


class PeaksAlignmentReward(Reward):

    """ Reward streamlines based on their alignment with local peaks
    and their past direction.
    """

    def __init__(
        self,
        peaks: MRIDataVolume,
        asymmetric: bool = False
    ):
        self.name = 'peaks_reward'

        self.peaks = peaks
        self.asymmetric = asymmetric

    def __call__(
        self,
        streamlines: torch.tensor,
        dones: torch.tensor
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
        N, L, _ = streamlines.shape

        if streamlines.shape[1] < 2:
            # Not enough segments to compute curvature
            return torch.ones(len(streamlines), dtype=torch.uint8)

        X, Y, Z, P = self.peaks.shape
        idx = streamlines[:, -2]

        # Get peaks at streamline end
        v = torch_nearest_interpolation(
            self.peaks, idx)

        # Presume 5 peaks (per hemisphere if asymmetric)
        if self.asymmetric:
            v = torch.reshape(v, (N, 5 * 2, P // (5 * 2)))
        else:
            v = torch.reshape(v, (N * 5, P // 5))

            # # Normalize peaks
            v = normalize_vectors(v)
            v = torch.reshape(v, (N, 5, P // 5))

            # Zero NaNs
            v = torch.nan_to_num(v)

        # Get last streamline segments

        dirs = torch.diff(streamlines, dim=1)
        u = dirs[:, -1]
        # Normalize segments
        u = normalize_vectors(u)

        # Zero NaNs
        u = torch.nan_to_num(u)

        # Get do product between all peaks and last streamline segments
        dot = torch.einsum('ijk,ik->ij', v, u)

        if not self.asymmetric:
            dot = torch.abs(dot)

        # Get alignment with the most aligned peak
        rewards = torch.amax(dot, dim=-1)
        # rewards = np.abs(dot)

        factors = torch.ones((N), device=streamlines.device)

        # Weight alignment with peaks with alignment to itself
        if streamlines.shape[1] >= 3:
            # Get previous to last segment
            w = dirs[:, -2]

            # # Normalize segments
            w = normalize_vectors(w)

            # # Zero NaNs
            w = torch.nan_to_num(w)

            # Calculate alignment between two segments
            factors = torch.einsum('ik,ik->i', u, w)

        # Penalize angle with last step
        rewards *= factors

        return rewards


class LengthReward(Reward):

    """ Reward streamlines based on their maximum and current length.
    """

    def __init__(
        self,
        max_length: int,
    ):
        """
        Parameters
        ----------
        max_length: int
            Maximum streamline length, in steps.
        """

        self.name = 'length_reward'

        self.max_length = max_length

    def __call__(
        self,
        streamlines: torch.tensor,
        dones: torch.tensor
    ):
        """
        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        dones: `numpy.ndarray` of shape (n_streamlines)
            Whether tracking is over for each streamline or not.

        Returns
        -------
        factor: torch.tensor of floats
            Reward components unweighted
        """
        N, S, _ = streamlines.shape

        factor = torch.full(
            (N,), S / self.max_length, device=streamlines.device)

        return factor


class TargetReward(Reward):

    """ Reward streamlines if they enter a "target mask" (GM).
    """

    def __init__(
        self,
        target: MRIDataVolume,
    ):
        """
        Parameters
        ----------
        target: torch.tensor
            Grey matter mask
        """

        self.name = 'target_reward'

        self.target = target.data,

    def __call__(
        self,
        streamlines: torch.tensor,
        dones: torch.tensor
    ):
        """
        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        dones: `numpy.ndarray` of shape (n_streamlines)
            Whether tracking is over for each streamline or not.

        Returns
        -------
        reward: torch.tensor of floats
            Reward components unweighted
        """

        target_streamlines = torch_trilinear_interpolation(
            streamlines, self.target, 0.9
        ).astype(int)

        factor = target_streamlines * dones * int(
            streamlines.shape[1] > self.min_nb_steps)

        return factor
