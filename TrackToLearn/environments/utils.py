import numpy as np
from dipy.tracking import metrics as tm
from dipy.tracking import utils as track_utils
from multiprocessing import Pool
from TrackToLearn.utils.utils import normalize_vectors


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


def is_too_long(
    streamlines: np.ndarray, bundles: np.ndarray, max_nb_steps: int
):
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


def is_too_curvy(
    streamlines: np.ndarray, bundles: np.ndarray, max_theta: float
):
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
        return np.zeros(streamlines.shape[0], dtype=bool)

    # Compute vectors for the last and before last streamline segments
    u = normalize_vectors(streamlines[:, -1] - streamlines[:, -2])
    v = normalize_vectors(streamlines[:, -2] - streamlines[:, -3])

    # Compute angles
    angles = np.arccos(np.einsum('ij,ij->i', u, v))
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

    clean_ids = remove_loops_and_sharp_turns(
        streamlines, loop_threshold, num_processes=8)
    mask = np.full(streamlines.shape[0], True)
    mask[clean_ids] = False
    return mask


def remove_loops_and_sharp_turns(streamlines,
                                 max_angle,
                                 num_processes=1):
    """
    Remove loops and sharp turns from a list of streamlines.
    Parameters
    ----------
    streamlines: list of ndarray
        The list of streamlines from which to remove loops and sharp turns.
    max_angle: float
        Maximal winding angle a streamline can have before
        being classified as a loop.
    use_qb: bool
        Set to True if the additional QuickBundles pass is done.
        This will help remove sharp turns. Should only be used on
        bundled streamlines, not on whole-brain tractograms.
    qb_threshold: float
        Quickbundles distance threshold, only used if use_qb is True.
    qb_seed: int
        Seed to initialize randomness in QuickBundles

    Returns
    -------
    list: the ids of clean streamlines
        Only the ids are returned so proper filtering can be done afterwards
    """

    ids = []
    pool = Pool(num_processes)
    windings = pool.map(tm.winding, streamlines)
    pool.close()
    ids = list(np.where(np.array(windings) < max_angle)[0])

    return ids


def seeds_from_head_tail(head_tail, affine, seed_count=1):
    """ Create seeds from a stack of head and tail masks

    Parameters
    ----------
    head_tail : `numpy.ndarray` of shape (x, y, z, n)
        Stack of head and tail masks
    affine : `numpy.ndarray` of shape (4, 4)
        Affine matrix to convert voxel coordinates to world coordinates
    seeds_count : int
        Number of seeds to generate per voxel for each slice

    Returns
    -------
    seeds : `numpy.ndarray`
        Tracking seeds
    bundle_idx: `numpy.ndarray`
        Corresponding bundle for all seeds
    """

    # For each slice of the head and tail masks, create seeds
    seeds = []
    bundle_idx = []
    for i in range(head_tail.shape[-1]):
        # Get the corresponding bundle mask
        ht_i = head_tail[..., i]
        # Generate seeds from it
        bundle_seeds = track_utils.random_seeds_from_mask(
            ht_i.astype(bool), affine, seeds_count=seed_count)
        # Add to list of seeds, keep track of the corresponding bundle
        seeds.extend(bundle_seeds)
        bundle_idx.extend(np.ones((bundle_seeds.shape[0])) * i)

    # Convert to np.ndarray for ease of handling
    seeds, bundle_idx = np.asarray(seeds), np.asarray(bundle_idx)
    # Shuffle to ensure proper coverage in batches
    idices = np.arange(seeds.shape[0])

    return seeds[idices], bundle_idx[idices]


def seeds_from_gm_atlas(atlas, interface, affine, seed_count=1):
    """  Create seeds from an atlas of gray matter regions.

    Parameters
    ----------
    atlas : `numpy.ndarray` of shape (x, y, z)
        Gray matter atlas with each region being a different integer
    interface : `numpy.ndarray` of shape (x, y, z)
        Interface mask
    affine : `numpy.ndarray` of shape (4, 4)
        Affine matrix to convert voxel coordinates to world coordinates
    seeds_count : int
        Number of seeds to generate per voxel for each slice.

    Returns
    -------
    seeds : `numpy.ndarray`
        Tracking seeds
    roi_idx: `numpy.ndarray`
        Corresponding ROI for all seeds
    """

    # Get the unique regions in the atlas
    regions = np.unique(atlas)
    # Remove 0 as it is not a region
    regions = regions[regions != 0]

    # For each region, create seeds
    seeds = []
    roi_idx = []
    for i, region in enumerate(regions):
        # Get the corresponding region mask
        region_mask = atlas == region
        # Generate seeds from it
        region_seeds = track_utils.random_seeds_from_mask(
            region_mask, affine, seeds_count=seed_count)
        # Add to list of seeds, keep track of the corresponding region
        seeds.extend(region_seeds)
        roi_idx.extend(np.ones((region_seeds.shape[0])) * i)

    # Convert to np.ndarray for ease of handling
    seeds, roi_idx = np.asarray(seeds), np.asarray(roi_idx)
    # Shuffle to ensure proper coverage in batches
    idices = np.arange(seeds.shape[0])

    return seeds[idices], roi_idx[idices]
