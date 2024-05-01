from enum import Enum

import numpy as np
from dipy.io.stateful_tractogram import Space, StatefulTractogram, Tractogram
from scipy.ndimage import map_coordinates, spline_filter

from TrackToLearn.environments.interpolation import (
    nearest_neighbor_interpolation)

from TrackToLearn.oracles.oracle import OracleSingleton
from TrackToLearn.utils.utils import normalize_vectors


class StoppingFlags(Enum):
    """ Predefined stopping flags to use when checking which streamlines
    should stop
    """
    STOPPING_MASK = int('00000001', 2)
    STOPPING_LENGTH = int('00000010', 2)
    STOPPING_CURVATURE = int('00000100', 2)
    STOPPING_TARGET = int('00001000', 2)
    STOPPING_LOOP = int('00010000', 2)
    STOPPING_ANGULAR_ERROR = int('00100000', 2)
    STOPPING_ORACLE = int('01000000', 2)


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


class BinaryStoppingCriterion(object):
    """
    Defines if a streamline is outside a mask using NN interp.
    """

    def __init__(
        self,
        mask: np.ndarray,
        threshold: float = 0.5,
    ):
        """
        Parameters
        ----------
        mask : 3D `numpy.ndarray`
            3D image defining a stopping mask. The interior of the mask is
            defined by values higher or equal than `threshold` .
        threshold : float
            Voxels with a value higher or equal than this threshold are
            considered as part of the interior of the mask.
        """
        self.mask = spline_filter(
            np.ascontiguousarray(mask, dtype=float), order=3)
        self.threshold = threshold

    def __call__(
        self,
        streamlines: np.ndarray,
    ):
        """ Checks which streamlines have their last coordinates outside a
        mask.

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
        coords = streamlines[:, -1, :].T
        return map_coordinates(
            self.mask, coords, prefilter=False
        ) < self.threshold


class AngularErrorCriterion(object):
    """ Defines if tracking should stop based on the maximum angular
    distance with the most aligned peak. This is to prevent streamlines
    from looping, as looping requires forgoing local directions for a
    while before tracking in "reverse".
    """

    def __init__(
        self,
        max_theta,
        peaks,
        asymmetric=False,
    ):
        """
        Parameters
        ----------
        max_theta: float
            Maximum angular distance between the streamline segment and
            the most aligned peak.
        peaks: 4D `numpy.ndarray`
            Local peaks.
        """

        self.max_theta_rad = np.deg2rad(max_theta)
        self.peaks = np.ascontiguousarray(peaks.data)
        self.asymmetric = False

    def __call__(
        self,
        streamlines: np.ndarray,
    ):
        """ Checks if the last streamline segment has an angular error below or
        above the limit.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        Returns
        -------
        angular_error_mask: 1D boolean `numpy.ndarray` of shape (n_streamlines)
            Array telling whether a streamline's last segment's angular error
            is above or below a threshold.
        """
        N, L, _ = streamlines.shape

        if streamlines.shape[1] < 2:
            # Not enough segments to compute curvature
            return np.ones(len(streamlines), dtype=np.uint8)

        X, Y, Z, P = self.peaks.shape
        idx = streamlines[:, -2].astype(np.int32)

        # Get peaks at streamline end
        v = nearest_neighbor_interpolation(
            self.peaks, idx)

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
        # Compute distance from cosine similarity
        distance = np.arccos(dot)
        # Get alignment with the most aligned peak
        min_distance = np.amin(distance, axis=-1)
        return min_distance > self.max_theta_rad


class OracleStoppingCriterion(object):
    """
    Defines if a streamline should stop according to the oracle.

    """

    def __init__(
        self,
        checkpoint: str,
        min_nb_steps: int,
        reference: str,
        affine_vox2rasmm: np.ndarray,
        device: str
    ):

        self.name = 'oracle_reward'

        if checkpoint:
            self.checkpoint = checkpoint
            self.model = OracleSingleton(checkpoint, device)
        else:
            self.checkpoint = None

        self.affine_vox2rasmm = affine_vox2rasmm
        self.reference = reference
        self.min_nb_steps = min_nb_steps
        self.device = device

    def __call__(
        self,
        streamlines: np.ndarray,
    ):
        """
        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space

        Returns
        -------
        dones: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
            Array indicating if streamlines are done.
        """
        if not self.checkpoint:
            return None

        # Resample streamlines to a fixed number of points. This should be
        # set by the model ? TODO?
        N, L, P = streamlines.shape
        if L > self.min_nb_steps:

            tractogram = Tractogram(
                streamlines=streamlines.copy())

            tractogram.apply_affine(self.affine_vox2rasmm)

            sft = StatefulTractogram(
                streamlines=tractogram.streamlines,
                reference=self.reference,
                space=Space.RASMM)

            sft.to_vox()
            sft.to_corner()
            predictions = self.model.predict(sft.streamlines)

            scores = np.zeros_like(predictions)
            scores[predictions < 0.5] = 1
            return scores.astype(bool)

        return np.array([False] * N)
