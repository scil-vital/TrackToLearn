import numpy as np

from numba import njit


@njit
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
    lower = np.asarray([0, 0, 0])
    upper = (np.asarray(volume.shape[:3]) - 1)
    indices = np.zeros_like(indices_unclipped, dtype=np.int32)
    indices[:, 0] = np.clip(indices_unclipped[:, 0], lower[0], upper[0])
    indices[:, 1] = np.clip(indices_unclipped[:, 1], lower[0], upper[1])
    indices[:, 2] = np.clip(indices_unclipped[:, 2], lower[0], upper[2])

    output = np.zeros((coords.shape[0], volume.shape[-1]))

    for i in range(coords.shape[0]):
        x, y, z = indices[i, 0], indices[i, 1], indices[i, 2]
        output[i] = volume[x][y][z]

    return output
