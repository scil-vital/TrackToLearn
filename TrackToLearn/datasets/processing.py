import numpy as np

from typing import Optional


def min_max_normalize_data_volume(
    data: np.ndarray,
    normalization_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """ Apply zero-centering and variance normalization to a data volume along each
    modality in the last axis (for voxels inside a given mask)

    Parameters:
    -----------
    data_sh : ndarray of shape (X, Y, Z, #modalities)
        Volume to normalize along each modality
    normalization_mask : binary ndarray of shape (X, Y, Z)
        3D mask defining which voxels should be used for normalization.
        If None, all non-zero voxels will be used.

    Returns
    -------
    normalized_data : ndarray of shape (X, Y, Z, #modalities)
        Normalized data volume, with zero-mean and unit variance along each
        axis of the last dimension
    """
    # Normalization in each direction (zero mean and unit variance)
    if normalization_mask is None:
        # If no mask is given, use non-zero data voxels
        normalization_mask = np.zeros(data.shape[:3], dtype=np.int32)
        nonzero_idx = np.nonzero(data.sum(axis=-1))
        normalization_mask[nonzero_idx] = 1
    else:
        # Mask resolution must fit DWI resolution
        assert normalization_mask.shape == data.shape[:3], \
                "Normalization mask resolution does not fit data..."

    normalized_data = data.copy()
    idx = np.nonzero(normalization_mask)
    v = normalized_data[idx]
    normalized_data[idx] = (v - v.min()) / (v.max() - v.min())

    return normalized_data
