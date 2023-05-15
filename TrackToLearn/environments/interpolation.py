import numpy as np
import torch

from scipy.ndimage.interpolation import map_coordinates


def torch_trilinear_interpolation(
    volume: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the data volume at given coordinates using trilinear
    interpolation on a torch tensor.

    Interpolation is done using the device on which the volume is stored.

    Parameters
    ----------
    volume : torch.Tensor with 3D or 4D shape
        The input volume to interpolate from
    coords : torch.Tensor with shape (N,3)
        The coordinates where to interpolate

    Returns
    -------
    output : torch.Tensor with shape (N, #modalities)
        The list of interpolated values

    References
    ----------
    [1] https://spie.org/samples/PM159.pdf
    """
    # Get device, and make sure volume and coords are using the same one
    assert volume.device == coords.device, "volume on device: {}; " \
                                           "coords on device: {}".format(
                                               volume.device,
                                               coords.device)
    coords = coords.type(torch.float32)
    volume = volume.type(torch.float32)

    device = volume.device

    B1_torch = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                             [-1, 0, 0, 0, 1, 0, 0, 0],
                             [-1, 0, 1, 0, 0, 0, 0, 0],
                             [-1, 1, 0, 0, 0, 0, 0, 0],
                             [1, 0, -1, 0, -1, 0, 1, 0],
                             [1, -1, -1, 1, 0, 0, 0, 0],
                             [1, -1, 0, 0, -1, 1, 0, 0],
                             [-1, 1, 1, -1, 1, -1, -1, 1]],
                            dtype=torch.float32, device=device)

    idx_torch = torch.tensor([[0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0],
                              [0, 1, 1],
                              [1, 0, 0],
                              [1, 0, 1],
                              [1, 1, 0],
                              [1, 1, 1]], dtype=torch.float32, device=device)

    if volume.dim() <= 2 or volume.dim() >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    if volume.dim() == 3:
        # torch needs indices to be cast to long
        indices_unclipped = (
            coords[:, None, :] + idx_torch).reshape((-1, 3)).long()

        # Clip indices to make sure we don't go out-of-bounds
        lower = torch.as_tensor([0, 0, 0]).to(device)
        upper = (torch.as_tensor(volume.shape) - 1).to(device)
        indices = torch.min(torch.max(indices_unclipped, lower), upper)

        # Fetch volume data at indices
        P = volume[
            indices[:, 0], indices[:, 1], indices[:, 2]
        ].reshape((coords.shape[0], -1)).t()

        d = coords - torch.floor(coords)
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        Q1 = torch.stack([
            torch.ones_like(dx), dx, dy, dz, dx * dy, dy * dz,
            dx * dz, dx * dy * dz],
            dim=0)
        output = torch.sum(P * torch.mm(B1_torch.t(), Q1), dim=0)

        return output

    if volume.dim() == 4:
        # 8 coordinates of the corners of the cube, for each input coordinate
        indices_unclipped = torch.floor(
            coords[:, None, :] + idx_torch).reshape((-1, 3)).long()

        # Clip indices to make sure we don't go out-of-bounds
        lower = torch.as_tensor([0, 0, 0], device=device)
        upper = torch.as_tensor(volume.shape[:3], device=device) - 1
        indices = torch.min(torch.max(indices_unclipped, lower), upper)

        # Fetch volume data at indices
        P = volume[indices[:, 0], indices[:, 1], indices[:, 2], :].reshape(
            (coords.shape[0], 8, volume.shape[-1]))

        # Shift 0.5 because fODFs are centered ?
        # coords = coords - 0.5
        d = coords - torch.floor(coords)
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        Q1 = torch.stack([
            torch.ones_like(dx), dx, dy, dz, dx * dy,
            dy * dz, dx * dz, dx * dy * dz],
            dim=0)
        output = torch.sum(
            P * torch.mm(B1_torch.t(), Q1).t()[:, :, None], dim=1)

        return output.type(torch.float32)

    raise ValueError(
        "There was a problem with the volume's number of dimensions!")


def interpolate_volume_at_coordinates(
    volume: np.ndarray,
    coords: np.ndarray,
    mode: str = 'nearest',
    order: int = 1,
    cval: float = 0.0
) -> np.ndarray:
    """ Evaluates a 3D or 4D volume data at the given coordinates using
    trilinear interpolation.

    Parameters
    ----------
    volume : 3D array or 4D array
        Data volume.
    coords : ndarray of shape (N, 3)
        3D coordinates where to evaluate the volume data.
    mode : str, optional
        Points outside the boundaries of the input are filled according to the
        given mode (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’).
        Default is ‘nearest’.
        ('constant' uses 0.0 as a points outside the boundary)

    Returns
    -------
    output : 2D array
        Values from volume.
    """
    # map_coordinates uses the center of the voxel, so should we shift to
    # the corner?

    if volume.ndim <= 2 or volume.ndim >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    if volume.ndim == 3:
        return map_coordinates(
            volume, coords.T, order=order, mode=mode, cval=cval)

    if volume.ndim == 4:
        D = volume.shape[-1]
        values_4d = np.zeros((coords.shape[0], D))
        for i in range(volume.shape[-1]):
            values_4d[:, i] = map_coordinates(
                volume[..., i], coords.T, order=order,
                mode=mode, cval=cval)
        return values_4d
