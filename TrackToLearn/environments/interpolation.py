import torch
import numpy as np

from dwi_ml.data.processing.space.neighborhood import \
    extend_coordinates_with_neighborhood

B1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
               [-1, 0, 0, 0, 1, 0, 0, 0],
               [-1, 0, 1, 0, 0, 0, 0, 0],
               [-1, 1, 0, 0, 0, 0, 0, 0],
               [1, 0, -1, 0, -1, 0, 1, 0],
               [1, -1, -1, 1, 0, 0, 0, 0],
               [1, -1, 0, 0, -1, 1, 0, 0],
               [-1, 1, 1, -1, 1, -1, -1, 1]], dtype=float)

# We will use the 8 voxels surrounding current position to interpolate a
# value. See ref https://spie.org/samples/PM159.pdf. The point p000 = [0, 0, 0]
# is the bottom corner of the current position (using floor).
idx_box = np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1]], dtype=float)


def torch_trilinear_interpolation(volume: torch.Tensor,
                                  coords_vox_corner: torch.Tensor):
    """Evaluates the data volume at given coordinates using trilinear
    interpolation on a torch tensor.

    Interpolation is done using the device on which the volume is stored.

    * Note. There is a function in torch:
    torch.nn.functional.interpolation with mode trilinear
    But it resamples volumes, not coordinates.

    Parameters
    ----------
    volume : torch.Tensor with 3D or 4D shape
        The input volume to interpolate from
    coords_vox_corner : torch.Tensor with shape (N,3)
        The coordinates where to interpolate. (Origin = corner, space = vox).

    Returns
    -------
    output : torch.Tensor with shape (N, #modalities)
        The list of interpolated values
    coords_to_idx_clipped: the coords after floor and clipping in box.

    References
    ----------
    [1] https://spie.org/samples/PM159.pdf
    """
    device = volume.device

    # Send data to device
    idx_box_torch = torch.as_tensor(idx_box, dtype=torch.float, device=device)
    B1_torch = torch.as_tensor(B1, dtype=torch.float, device=device)

    if volume.dim() <= 2 or volume.dim() >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    # - indices are the floor of coordinates + idx, boxes with 8 corners around
    #   given coordinates. (Floor means origin = corner)
    # - coords + idx_torch shape -> the box of 8 corners around each coord
    #   reshaped as (-1,3) = [n * 8, 3]
    # - torch needs indices to be cast to long
    # - clip indices to make sure we don't go out-of-bounds
    #   Origin = corner means the minimum is 0.
    #                         the maximum is shape.
    # Ex, for shape 150, last voxel is #149, with possible coords up to 149.99.
    lower = torch.as_tensor([0, 0, 0], device=device)
    upper = torch.as_tensor(volume.shape[:3], device=device) - 1
    idx_box_clipped = torch.min(
        torch.max(
            torch.floor(coords_vox_corner[:, None, :] + idx_box_torch
                        ).reshape((-1, 3)).long(),
            lower),
        upper)

    # Setting Q1 such as in equation 9.9
    d = coords_vox_corner - torch.floor(coords_vox_corner)
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    Q1 = torch.stack([torch.ones_like(dx), dx, dy, dz,
                      dx * dy, dy * dz, dx * dz,
                      dx * dy * dz], dim=0)

    # As of now:
    # B1 = 8x8
    # Q1 = 8 x n (GROS)
    # mult B1 * Q1 = 8 x n
    # overwriting Q1 with mult to try and save space
    if volume.dim() == 3:
        Q1 = torch.mm(B1_torch.t(), Q1)

        # Fetch volume data at indices based on equation 9.11.
        p = volume[idx_box_clipped[:, 0],
                   idx_box_clipped[:, 1],
                   idx_box_clipped[:, 2]]
        # Last dim (-1) = the 8 corners
        p = p.reshape((coords_vox_corner.shape[0], -1)).t()

        # Finding coordinates with equation 9.12a.
        return torch.sum(p * Q1, dim=0)

    elif volume.dim() == 4:
        Q1 = torch.mm(B1_torch.t(), Q1).t()[:, :, None]

        # Fetch volume data at indices
        p = volume[idx_box_clipped[:, 0],
                   idx_box_clipped[:, 1],
                   idx_box_clipped[:, 2], :]
        p = p.reshape((coords_vox_corner.shape[0], 8, volume.shape[-1]))

        # p: of shape n x 8 x features
        # Q1: n x 8 x 1

        # return torch.sum(p * Q1, dim=1)
        # Able to have bigger batches by avoiding 3D matrix.
        # Ex: With neighborhood axis [1 2] (13 neighbors), 47 features per
        # point, we can pass from batches of 1250 streamlines to 2300!
        total = torch.zeros(p.shape[0], p.shape[2], device=device,
                            dtype=torch.float)
        for corner in range(8):
            total += p[:, corner, :] * Q1[:, corner, :]
        return total

    else:
        raise ValueError("Interpolation: There was a problem with the "
                         "volume's number of dimensions!")


# @njit
def nearest_neighbor_interpolation(
    volume: np.array([[[[]]]]),
    coords: np.ndarray,
    cval: float = 0
) -> np.ndarray:
    """ Get the nearest neighbor interpolation of a 3/4D volume, where
    the output will be of shape (N, D) with N the number of coordinates
    and N the length of the last dimension of the volume.

    Presumes coordinates are using origin=corner.
    """
    coords = coords
    volume = volume

    if volume.ndim <= 2 or volume.ndim >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    indices_unclipped = np.floor(coords).astype(np.int32)

    # Clip indices to make sure we don't go out-of-bounds
    upper = (np.asarray(volume.shape[:3]) - 1)

    indices = np.clip(indices_unclipped, 0, upper).astype(int)
    output = volume[tuple(indices.T)]

    check = ~np.all(np.equal(indices_unclipped, indices), axis=1)
    if volume.ndim == 4:
        output[check] = np.ones((sum(check), output.shape[-1])) * cval
    else:
        output[check] = np.ones((sum(check))) * cval

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
