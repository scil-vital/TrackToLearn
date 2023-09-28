import numpy as np

from dipy.io.stateful_tractogram import StatefulTractogram, Space, Tractogram
from enum import Enum
from scipy.ndimage import spline_filter, map_coordinates

from TrackToLearn.environments.interpolation import (
    interpolate_volume_at_coordinates,
    nearest_neighbor_interpolation)

from TrackToLearn.oracles.oracle import OracleSingleton
from TrackToLearn.utils.utils import normalize_vectors


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
        self.mask = spline_filter(np.ascontiguousarray(mask), order=3)
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
        return map_coordinates(
            self.mask, streamlines[:, -1, :].T, prefilter=False
        ) < self.threshold
        # # Get last streamlines coordinates
        # return interpolate_volume_at_coordinates(
        #     self.mask, streamlines[:, -1, :], mode='constant',
        #     order=3, filter=) < self.threshold


class CmcStoppingCriterion(object):
    """ Checks which streamlines should stop according to Continuous map
    criteria.
    Ref:
        Girard, G., Whittingstall, K., Deriche, R., & Descoteaux, M. (2014)
        Towards quantitative connectivity analysis: reducing tractography
        biases.
        Neuroimage, 98, 266-278.

    This was included in the abstract
        Theberge, A., Poirier, C., Petit, L., Jodoin, P.-M., Descoteaux, M., (2022)
        Incorporating Anatomical Priors into Track-to-Learn. ISMRM Diffusion
        Workshop: from Research to Clinic.

    This is only in the partial-spirit of CMC. A good improvement (#TODO)
    would be to include or exclude streamlines from the resulting
    tractogram as well. Let me know if you need help in adding this
    functionnality.
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
        affine_vox2rasmm: `numpy.ndarray` with shape (4,4) (optional)
            Tranformation that aligns brings streamlines to rasmm from vox.
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
        """
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
            order=3)
        if streamlines.shape[1] < self.min_nb_steps:
            include_result[:] = 0.

        exclude_result = interpolate_volume_at_coordinates(
            self.exclude_mask, streamlines[:, -1, :], mode='constant',
            order=3, cval=1.0)

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

            # TODO: What the actual fuck
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
            scores[predictions <= 0.5] = 1.0
            return scores.astype(bool)

        return np.array([False] * N)
