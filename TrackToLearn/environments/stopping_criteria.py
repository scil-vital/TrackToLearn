import numpy as np

from enum import Enum

from TrackToLearn.environments.utils import interpolate_volume_at_coordinates


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
        self.mask = mask
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

        # Get last streamlines coordinates
        return interpolate_volume_at_coordinates(
            self.mask, streamlines[:, -1, :], mode='constant',
            order=0) < self.threshold


class CmcStoppingCriterion(object):
    """ Checks which streamlines should stop according to Continuous map
    criteria.
    Ref:
        Girard, G., Whittingstall, K., Deriche, R., & Descoteaux, M. (2014)
        Towards quantitative connectivity analysis: reducing tractography
        biases.
        Neuroimage, 98, 266-278.

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
