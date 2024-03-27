from enum import Enum

import numpy as np
from dipy.io.stateful_tractogram import Space, StatefulTractogram, Tractogram
from scipy.ndimage import map_coordinates, spline_filter

from TrackToLearn.oracles.oracle import OracleSingleton


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
        coords = streamlines[:, -1, :].T - 0.5
        return map_coordinates(
            self.mask, coords, prefilter=False
        ) < self.threshold


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
