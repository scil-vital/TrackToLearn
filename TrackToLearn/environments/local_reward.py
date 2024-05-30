import numpy as np

from TrackToLearn.environments.interpolation import (
    nearest_neighbor_interpolation)
from TrackToLearn.datasets.utils import MRIDataVolume
from TrackToLearn.environments.reward import Reward
from TrackToLearn.utils.utils import normalize_vectors


class PeaksAlignmentReward(Reward):

    """ Reward streamlines based on their alignment with local peaks
    and their past direction.

    Initially proposed in
        Théberge, A., Desrosiers, C., Descoteaux, M., & Jodoin, P. M. (2021).
        Track-to-learn: A general framework for tractography with deep
        reinforcement learning. Medical Image Analysis, 72, 102093.
    """

    def __init__(
        self,
        peaks: MRIDataVolume,
    ):
        self.name = 'peaks_reward'

        self.peaks = peaks.data

    def __call__(
        self,
        streamlines: np.ndarray,
        bundle_idx: np.ndarray,
        dones: np.ndarray
    ):
        """
        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        bundle_idx : `numpy.ndarray` of shape (n_streamlines,)
            Bundle index of each streamline
        dones : `numpy.ndarray` of shape (n_streamlines,)
            Whether each streamline has reached the end of the episode

        Returns
        -------
        rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
            Array containing the reward
        """
        N, L, _ = streamlines.shape

        if streamlines.shape[1] < 2:
            # Not enough segments to compute curvature
            return np.ones(len(streamlines), dtype=np.uint8)

        X, Y, Z, P = self.peaks.shape
        idx = streamlines[:, -2].astype(np.int32)

        # Get peaks at streamline end
        v = nearest_neighbor_interpolation(self.peaks, idx)

        v = np.reshape(v, (N * 5, P // 5))

        with np.errstate(divide='ignore', invalid='ignore'):
            # # Normalize peaks
            v = normalize_vectors(v)

        v = np.reshape(v, (N, 5, P // 5))
        # Zero NaNs
        v = np.nan_to_num(v)

        # Get last streamline segments

        dirs = np.diff(streamlines, axis=1)
        u = dirs[:, -1]
        # Normalize segments
        with np.errstate(divide='ignore', invalid='ignore'):
            u = normalize_vectors(u)

        # Zero NaNs
        u = np.nan_to_num(u)

        # Get do product between all peaks and last streamline segments
        dot = np.einsum('ijk,ik->ij', v, u)

        dot = np.abs(dot)

        # Get alignment with the most aligned peak
        rewards = np.amax(dot, axis=-1)
        # rewards = np.abs(dot)

        factors = np.ones((N))

        # Weight alignment with peaks with alignment to itself
        if streamlines.shape[1] >= 3:
            # Get previous to last segment
            w = dirs[:, -2]

            # # Normalize segments
            with np.errstate(divide='ignore', invalid='ignore'):
                w = normalize_vectors(w)

            # # Zero NaNs
            w = np.nan_to_num(w)

            # Calculate alignment between two segments
            np.einsum('ik,ik->i', u, w, out=factors)

        # Penalize angle with last step
        rewards *= factors

        return rewards
