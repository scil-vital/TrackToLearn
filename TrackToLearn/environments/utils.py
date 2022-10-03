import numpy as np
import torch

from enum import Enum

from scipy.ndimage.interpolation import map_coordinates

from TrackToLearn.utils.utils import normalize_vectors


B1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
               [-1, 0, 0, 0, 1, 0, 0, 0],
               [-1, 0, 1, 0, 0, 0, 0, 0],
               [-1, 1, 0, 0, 0, 0, 0, 0],
               [1, 0, -1, 0, -1, 0, 1, 0],
               [1, -1, -1, 1, 0, 0, 0, 0],
               [1, -1, 0, 0, -1, 1, 0, 0],
               [-1, 1, 1, -1, 1, -1, -1, 1]], dtype=np.float)

idx = np.array([[0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1]], dtype=np.float)


# Flags enum
class StoppingFlags(Enum):
    """ Predefined stopping flags to use when checking which streamlines
    should stop
    """
    STOPPING_MASK = int('00000001', 2)
    STOPPING_LENGTH = int('00000010', 2)
    STOPPING_CURVATURE = int('00000100', 2)
    STOPPING_TARGET = int('00001000', 2)
    STOPPING_LOOP = int('00010000', 2)


def get_sh(
    segments,
    data_volume,
    add_neighborhood_vox,
    neighborhood_directions,
    history,
    device
) -> np.ndarray:
    """ Get the sh coefficients at the end of streamlines
    """

    N, H, P = segments.shape
    flat_coords = np.reshape(segments, (N * H, P))

    coords = torch.as_tensor(flat_coords, device=device)
    n_coords = coords.shape[0]

    if add_neighborhood_vox:
        # Extend the coords array with the neighborhood coordinates
        coords = torch.repeat_interleave(
            coords,
            neighborhood_directions.size()[0],
            axis=0)

        coords[:, :3] += \
            neighborhood_directions.repeat(n_coords, 1)

        # Evaluate signal as if all coords were independent
        partial_signal = torch_trilinear_interpolation(
            data_volume, coords)

        # Reshape signal into (n_coords, new_feature_size)
        new_feature_size = partial_signal.size()[-1] * \
            neighborhood_directions.size()[0]
    else:
        partial_signal = torch_trilinear_interpolation(
            data_volume,
            coords).type(torch.float32)
        new_feature_size = partial_signal.size()[-1]

    signal = torch.reshape(partial_signal, (N, history * new_feature_size))

    assert len(signal.size()) == 2, signal.size()

    return signal


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
    local_idx = idx[:]

    # Send data to device
    idx_torch = torch.as_tensor(local_idx, dtype=torch.float, device=device)
    B1_torch = torch.as_tensor(B1, dtype=torch.float, device=device)

    if volume.dim() <= 2 or volume.dim() >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    if volume.dim() == 3:
        # torch needs indices to be cast to long
        indices_unclipped = (
            coords[:, None, :] + idx_torch).reshape((-1, 3)).long()

        # Clip indices to make sure we don't go out-of-bounds
        lower = torch.as_tensor([0, 0, 0], device=device)
        upper = torch.as_tensor(volume.shape, device=device) - 1
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
    cval: float = 0.0,
) -> np.ndarray:
    """ Evaluates a 3D or 4D volume data at the given coordinates using trilinear
    interpolation.

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
                volume[..., i], coords.T, order=order, mode=mode, cval=cval)
        return values_4d


def get_neighborhood_directions(
    radius: float
) -> np.ndarray:
    """ Returns predefined neighborhood directions at exactly `radius` length
        For now: Use the 6 main axes as neighbors directions, plus (0,0,0)
        to keep current position

    Parameters
    ----------
    radius : float
        Distance to neighbors

    Returns
    -------
    directions : `numpy.ndarray` with shape (n_directions, 3)

    Notes
    -----
    Coordinates are in voxel-space
    """
    axes = np.identity(3)
    directions = np.concatenate(([[0, 0, 0]], axes, -axes)) * radius
    return directions


def has_reached_gm(
    streamlines: np.ndarray,
    mask: np.ndarray,
    affine_vox2mask: np.ndarray = None,
    threshold: float = 0.,
    min_nb_steps: int = 10
):
    """ Checks which streamlines have their last coordinates inside a mask and
    are at least longer than a minimum strealine length.

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    mask : 3D `numpy.ndarray`
        3D image defining a stopping mask. The interior of the mask is defined
        by values higher or equal than `threshold` .
        NOTE: The mask coordinates can be in a different space than the
        streamlines coordinates if an affine is provided.
    affine_vox2mask : `numpy.ndarray` with shape (4,4) (optional)
        Tranformation that aligns streamlines on top of `mask`.
    threshold : float
        Voxels with a value higher or equal than this threshold are considered
        as part of the interior of the mask.
    min_length: float
        Minimum streamline length to end

    Returns
    -------
    inside : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline's can end after reaching GM.
    """
    return np.logical_and(is_inside_mask(
        streamlines, mask, affine_vox2mask, threshold),
        np.full(streamlines.shape[0], streamlines.shape[1] > min_nb_steps))


def is_inside_mask(
    streamlines: np.ndarray,
    mask: np.ndarray,
    affine_vox2mask: np.ndarray = None,
    threshold: float = 0.
):
    """ Checks which streamlines have their last coordinates inside a mask.

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    mask : 3D `numpy.ndarray`
        3D image defining a stopping mask. The interior of the mask is defined
        by values higher or equal than `threshold` .
        NOTE: The mask coordinates can be in a different space than the
        streamlines coordinates if an affine is provided.
    affine_vox2mask : `numpy.ndarray` with shape (4,4) (optional)
        Tranformation that aligns streamlines on top of `mask`.
    threshold : float
        Voxels with a value higher or equal than this threshold are considered
        as part of the interior of the mask.

    Returns
    -------
    inside : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline's last coordinate is inside the mask
        or not.
    """
    # Get last streamlines coordinates
    return interpolate_volume_at_coordinates(
        mask, streamlines[:, -1, :], mode='constant', order=0) >= threshold


class BinaryStoppingCriterion(object):
    """
    Defines if a streamline is outside a mask using trilinear interp.
    """

    def __init__(
        self,
        mask: np.ndarray,
        affine: np.ndarray = None,
        threshold: float = 0.5,
    ):
        """
        Parameters
        ----------

        mask : 3D `numpy.ndarray`
            3D image defining a stopping mask. The interior of the mask is
            defined by values higher or equal than `threshold` .
            NOTE: The mask coordinates can be in a different space than the
            streamlines coordinates if an affine is provided.
        affine_vox2mask : `numpy.ndarray` with shape (4,4) (optional)
            Tranformation that aligns streamlines on top of `mask`.
        threshold : float
            Voxels with a value higher or equal than this threshold are
            considered as part of the interior of the mask.

        """
        self.mask = mask
        self.affine = affine
        self.threshold = threshold

    def __call__(
        self,
        streamlines: np.ndarray,
    ):
        """ Checks which streamlines have their last coordinates outside a mask.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space

        Returns
        -------
        outside : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
            Array telling whether a streamline's last coordinate is outside the
            mask or not.
        """

        # Get last streamlines coordinates
        return interpolate_volume_at_coordinates(
            self.mask, streamlines[:, -1, :], mode='constant',
            order=0) < self.threshold


class CmcStoppingCriterion(object):
    """
    Defines if a streamline is outside a mask using trilinear interp.
    """

    def __init__(
        self,
        include_mask: np.ndarray,
        exclude_mask: np.ndarray,
        affine: np.ndarray,
        step_size: float,
        min_nb_steps: int,
    ):
        """
        Parameters
        ----------

        mask : 3D `numpy.ndarray`
            3D image defining a stopping mask. The interior of the mask is
            defined by values higher or equal than `threshold` .
            NOTE: The mask coordinates can be in a different space than the
            streamlines coordinates if an affine is provided.
        affine_vox2mask : `numpy.ndarray` with shape (4,4) (optional)
            Tranformation that aligns streamlines on top of `mask`.
        threshold : float
            Voxels with a value higher or equal than this threshold are
            considered as part of the interior of the mask.

        """
        self.include_mask = include_mask
        self.exclude_mask = exclude_mask
        self.affine = affine
        vox_size = np.mean(np.abs(np.diag(affine)[:3]))
        self.correction_factor = step_size / vox_size
        self.min_nb_steps = min_nb_steps

    def __call__(
        self,
        streamlines: np.ndarray,
    ):
        """ Checks which streamlines should stop according to Continuous map
        criteria.

        Ref:
            Girard, G., Whittingstall, K., Deriche, R., & Descoteaux, M. (2014)
            Towards quantitative connectivity analysis: reducing tractography
            biases.
            Neuroimage, 98, 266-278.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space

        Returns
        -------
        outside : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
            Array telling whether a streamline's last coordinate is outside the
            mask or not.
        """

        include_result = interpolate_volume_at_coordinates(
            self.include_mask, streamlines[:, -1, :], mode='constant',
            order=1)
        if streamlines.shape[1] < self.min_nb_steps:
            include_result[:] = 0.

        exclude_result = interpolate_volume_at_coordinates(
            self.exclude_mask, streamlines[:, -1, :], mode='constant',
            order=1, cval=1.0)

        # If streamlines are still in 100% WM, don't exit
        wm_points = include_result + exclude_result <= 0

        # Compute continue probability
        num = np.maximum(0, (1 - include_result - exclude_result))
        den = num + include_result + exclude_result
        p = (num / den) ** self.correction_factor

        # p >= continue prob -> not continue
        not_continue_points = np.random.random(streamlines.shape[0]) >= p

        # if by some magic some wm point don't continue, make them continue
        not_continue_points[wm_points] = False

        # if the point is in the include map, it has potentially reached GM
        p = (include_result / (include_result + exclude_result))
        stop_include = np.random.random(streamlines.shape[0]) < p
        not_continue_points[stop_include] = True

        return not_continue_points


def is_too_long(streamlines: np.ndarray, max_nb_steps: int):
    """ Checks whether streamlines have exceeded the maximum number of steps

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    max_nb_steps : int
        Maximum number of steps a streamline can have

    Returns
    -------
    too_long : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline is too long or not
    """
    return np.full(streamlines.shape[0], streamlines.shape[1] >= max_nb_steps)


def is_too_curvy(streamlines: np.ndarray, max_theta: float):
    """ Checks whether streamlines have exceeded the maximum angle between the
    last 2 steps

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    max_theta : float
        Maximum angle in degrees that two consecutive segments can have between
        each other.

    Returns
    -------
    too_curvy : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline is too curvy or not
    """
    max_theta_rad = np.deg2rad(max_theta)  # Internally use radian
    if streamlines.shape[1] < 3:
        # Not enough segments to compute curvature
        return np.zeros(streamlines.shape[0], dtype=np.uint8)

    # Compute vectors for the last and before last streamline segments
    u = normalize_vectors(streamlines[:, -1] - streamlines[:, -2])
    v = normalize_vectors(streamlines[:, -2] - streamlines[:, -3])

    # Compute angles
    angles = np.arccos(np.sum(u * v, axis=1).clip(-1., 1.))

    return angles > max_theta_rad


def winding(nxyz: np.ndarray) -> np.ndarray:
    """ Project lines to best fitting planes. Calculate
    the cummulative signed angle between each segment for each line
    and their previous one

    Adapted from dipy.tracking.metrics.winding to allow multiple
    lines that have the same length

    Parameters
    ------------
    nxyz : np.ndarray of shape (N, M, 3)
        Array representing x,y,z of M points in N tracts.

    Returns
    ---------
    a : np.ndarray
        Total turning angle in degrees for all N tracts.
    """

    directions = np.diff(nxyz, axis=1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    thetas = np.einsum(
        'ijk,ijk->ij', directions[:, :-1], directions[:, 1:]).clip(-1., 1.)
    shape = thetas.shape
    rads = np.arccos(thetas.flatten())
    turns = np.sum(np.reshape(rads, shape), axis=-1)
    return np.rad2deg(turns)

    # # This is causing a major slowdown :(
    # U, s, V = np.linalg.svd(nxyz-np.mean(nxyz, axis=1, keepdims=True), 0)

    # Up = U[:, :, 0:2]
    # # Has to be a better way than iterare over all tracts
    # diags = np.stack([np.diag(sp[0:2]) for sp in s], axis=0)
    # proj = np.einsum('ijk,ilk->ijk', Up, diags)

    # v0 = proj[:, :-1]
    # v1 = proj[:, 1:]
    # v = np.einsum('ijk,ijk->ij', v0, v1) / (
    #     np.linalg.norm(v0, axis=-1, keepdims=True)[..., 0] *
    #     np.linalg.norm(v1, axis=-1, keepdims=True)[..., 0])
    # np.clip(v, -1, 1, out=v)
    # shape = v.shape
    # rads = np.arccos(v.flatten())
    # turns = np.sum(np.reshape(rads, shape), axis=-1)

    # return np.rad2deg(turns)


def is_looping(streamlines: np.ndarray, loop_threshold: float):
    """ Checks whether streamlines are looping

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    looping_threshold: float
        Maximum angle in degrees for the whole streamline

    Returns
    -------
    too_curvy : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline is too curvy or not
    """

    angles = winding(streamlines)

    return angles > loop_threshold


def is_flag_set(flags, ref_flag):
    """ Checks which flags have the `ref_flag` set. """
    if type(ref_flag) is StoppingFlags:
        ref_flag = ref_flag.value
    return ((flags.astype(np.uint8) & ref_flag) >>
            np.log2(ref_flag).astype(np.uint8)).astype(bool)


def count_flags(flags, ref_flag):
    """ Counts how many flags have the `ref_flag` set. """
    if type(ref_flag) is StoppingFlags:
        ref_flag = ref_flag.value
    return is_flag_set(flags, ref_flag).sum()


def format_state(
    streamlines: np.ndarray,
    data_volume,
    add_neighborhood_vox,
    neighborhood_directions,
    n_signal,
    n_dirs,
    device
) -> np.ndarray:
    """
    From the last streamlines coordinates, extract the corresponding
    SH coefficients

    Parameters
    ----------
    streamlines: `numpy.ndarry`
        Streamlines from which to get the coordinates

    Returns
    -------
    signal: `numpy.ndarray`
        SH coefficients at the coordinates
    """
    N, L, P = streamlines.shape
    if N <= 0:
        return []
    segments = streamlines[:, -1, :][:, None, :]
    signal = get_sh(
        segments,
        data_volume,
        add_neighborhood_vox,
        neighborhood_directions,
        n_signal,
        device
    ).cpu().numpy()

    N, S = signal.shape

    inputs = np.zeros((N, S + (n_dirs * P)))

    inputs[:, :S] = signal

    previous_dirs = np.zeros((N, n_dirs, P), dtype=np.float32)
    if L > 1:
        dirs = np.diff(streamlines, axis=1)
        previous_dirs[:, :min(dirs.shape[1], n_dirs), :] = \
            dirs[:, :-(n_dirs+1):-1, :]

    dir_inputs = np.reshape(previous_dirs, (N, n_dirs * P))

    inputs[:, S:] = dir_inputs
    return inputs
