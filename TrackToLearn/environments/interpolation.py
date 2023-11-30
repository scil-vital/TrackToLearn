import numpy as np

from numba import njit


# @njit
def nearest_neighbor_interpolation(
    volume: np.array([[[[]]]]),
    coords: np.ndarray,
) -> np.ndarray:
    """
    """
    coords = coords
    volume = volume

    if volume.ndim <= 3 or volume.ndim >= 5:
        raise ValueError("Volume must be 4D!")

    indices_unclipped = np.round(coords).astype(np.int32)

    # Clip indices to make sure we don't go out-of-bounds
    upper = (np.asarray(volume.shape[:3]) - 1)
    indices = np.clip(indices_unclipped, 0, upper).astype(int).T
    output = volume[tuple(indices)]

    return output
