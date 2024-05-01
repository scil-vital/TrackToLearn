import numpy as np

# from numba import njit

from dwi_ml.data.processing.space.neighborhood import \
    extend_coordinates_with_neighborhood

from dwi_ml.data.processing.volume.interpolation import \
    torch_trilinear_interpolation


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


def interpolate_volume_in_neighborhood(
        volume_as_tensor, coords_vox_corner, neighborhood_vectors_vox=None):
    """
    Params
    ------
    data_tensor: tensor
        The data: a 4D tensor with last dimension F (nb of features).
    coords_vox_corner: torch.Tensor shape (M, 3)
        A list of points (3d coordinates). Neighborhood will be added to these
        points based. Coords must be in voxel world, origin='corner', to use
        trilinear interpolation.
    neighborhood_vectors_vox: np.ndarray[float] with shape (N, 3)
        The neighboors to add to each coord. Do not include the current point
        ([0,0,0]). Values are considered in the same space as
        coords_vox_corner, and should thus be in voxel space.

    Returns
    -------
    subj_x_data: tensor of shape (M, F * (N+1))
        The interpolated data: M points with contatenated neighbors.
    coords_vox_corner: tensor of shape (M x (N+1), 3)
        The final coordinates.
    """
    if (neighborhood_vectors_vox is not None and
            len(neighborhood_vectors_vox) > 0):
        m_input_points = coords_vox_corner.shape[0]
        n_neighb = neighborhood_vectors_vox.shape[0]
        f_features = volume_as_tensor.shape[-1]

        # Extend the coords array with the neighborhood coordinates
        # coords: shape (M x (N+1), 3)
        coords_vox_corner, tiled_vectors = \
            extend_coordinates_with_neighborhood(coords_vox_corner,
                                                 neighborhood_vectors_vox)

        # Interpolate signal for each (new) point
        # DWI data features for each neighbor are concatenated.
        # Result is of shape: (M * (N+1), F).
        flat_subj_x_data = torch_trilinear_interpolation(volume_as_tensor,
                                                         coords_vox_corner)

        # Neighbors become new features of the current point.
        # Reshape signal into (M, (N+1)*F))
        new_nb_features = f_features * n_neighb
        subj_x_data = flat_subj_x_data.reshape(m_input_points, new_nb_features)

    else:  # No neighborhood:
        subj_x_data = torch_trilinear_interpolation(volume_as_tensor,
                                                    coords_vox_corner)

    return subj_x_data, coords_vox_corner
