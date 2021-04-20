import numpy as np

from TrackToLearn.environments.utils import interpolate_volume_at_coordinates
from TrackToLearn.datasets.utils import (
    MRIDataVolume)


def reward_streamlines_step(
    streamlines: np.ndarray,
    peaks: MRIDataVolume,
    exclude: MRIDataVolume,
    target: MRIDataVolume,
    max_len: float,
    max_angle: float,
    affine_vox2mask: np.ndarray = None,
) -> list:

    """
    Compute rewards for the last step of the streamlines
    Each reward component is weighted according to a
    coefficient

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    peaks: `MRIDataVolume`
        Volume containing the fODFs peaks
    target_mask: MRIDataVolume
        Mask representing the tracking endpoints
    exclude_mask: MRIDataVolume
        Mask representing the tracking no-go zones
    max_len: `float`
        Maximum lengths for the streamlines (in terms of points)
    max_angle: `float`
        Maximum degrees between two streamline segments
    affine_vox2mask: np.ndarray
        Affine for moving stuff to voxel space

    Returns
    -------
    rewards: `float`
        Reward components weighted by their coefficients as well
        as the penalites
    """
    rewards = reward_alignment_with_peaks(
        streamlines, peaks.data, affine_vox2mask) \

    return rewards


def reward_alignment_with_peaks(
    streamlines, peaks, affine_vox2mask
):
    """ Reward streamlines according to the alignment to their corresponding
        fODFs peaks

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

    X, Y, Z, P = peaks.shape
    idx = streamlines[:, -2].astype(np.int)

    dirs = np.diff(streamlines, axis=1)

    # Get peaks at streamline end
    v = interpolate_volume_at_coordinates(
        peaks, idx, mode='constant', order=0)
    v = np.reshape(v, (N, 5, P // 5))

    with np.errstate(divide='ignore', invalid='ignore'):
        # # Normalize peaks
        v = v / np.sqrt(np.sum(v ** 2, axis=-1, keepdims=True))
    # Zero NaNs
    v = np.nan_to_num(v)

    # Get last streamline segments
    u = dirs[:, -1]
    # Normalize segments
    with np.errstate(divide='ignore', invalid='ignore'):
        u = u / np.sqrt(np.sum(u ** 2, axis=-1, keepdims=True))
    # Zero NaNs
    u = np.nan_to_num(u)

    # Get do product between all peaks and last streamline segments
    dot = np.einsum('ijk,ik->ij', v, u)

    # Get alignment with the most aligned peak
    rewards = np.amax(np.abs(dot), axis=-1)
    # rewards = np.abs(dot)

    factors = np.ones((N))

    # Weight alignment with peaks with alignment to itself
    if streamlines.shape[1] >= 3:
        # Get previous to last segment
        w = dirs[:, -2]

        # # Normalize segments
        with np.errstate(divide='ignore', invalid='ignore'):
            w = w / np.sqrt(np.sum(w ** 2, axis=-1, keepdims=True))

        # # Zero NaNs
        w = np.nan_to_num(w)

        # Calculate alignment between two segments
        factors = np.einsum('ik,ik->i', u, w)

    # Penalize angle with last step
    computed_rewards = rewards * factors

    return computed_rewards
