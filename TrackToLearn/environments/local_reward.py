import numpy as np

from TrackToLearn.environments.utils import (
    interpolate_volume_at_coordinates,
    is_inside_mask)
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

        self.peaks = peaks.data
        self.asymmetric = asymmetric

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
        N, L, _ = streamlines.shape

        if streamlines.shape[1] < 2:
            # Not enough segments to compute curvature
            return np.ones(len(streamlines), dtype=np.uint8)

        X, Y, Z, P = self.peaks.shape
        idx = streamlines[:, -2].astype(np.int32)

        # Get peaks at streamline end
        v = interpolate_volume_at_coordinates(
            self.peaks, idx, mode='nearest', order=0)

        # Presume 5 peaks (per hemisphere if asymmetric)
        if self.asymmetric:
            v = np.reshape(v, (N, 5 * 2, P // (5 * 2)))
        else:
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

        if not self.asymmetric:
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
        streamlines: np.ndarray,
        dones: np.ndarray
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
        factor: np.ndarray of floats
            Reward components unweighted
        """
        N, S, _ = streamlines.shape

        factor = np.asarray([S] * N) / self.max_length

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
        target: np.ndarray
            Grey matter mask
        """

        self.name = 'target_reward'

        self.target = target.data,

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
        dones: `numpy.ndarray` of shape (n_streamlines)
            Whether tracking is over for each streamline or not.

        Returns
        -------
        reward: np.ndarray of floats
            Reward components unweighted
        """

        target_streamlines = is_inside_mask(
            streamlines, self.target, 0.5
        ).astype(int)

        factor = target_streamlines * dones * int(
            streamlines.shape[1] > self.min_nb_steps)

        return factor
