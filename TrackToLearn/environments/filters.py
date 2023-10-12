import numpy as np
from enum import Enum

from dipy.io.stateful_tractogram import Space, StatefulTractogram, Tractogram

from TrackToLearn.oracles.oracle import OracleSingleton


class Filters(Enum):
    """ Predefined stopping flags to use when checking which streamlines
    should stop
    """
    CMC = 1
    MIN_LENGTH = 2
    ORACLE = 3


class CMCFilter:
    """ TODO
    """

    def _init__(
        self,
        include_mask: np.ndarray,
        exclude_mask: np.ndarray,
        affine: np.ndarray,
        step_size: float,
        min_nb_steps: int,
    ):
        pass


class MinLengthFilter:
    """ TODO
    """

    def __init__(
        self,
        min_nb_steps: int,
    ):

        self.name = 'min_length_filter'

        self.min_nb_steps = min_nb_steps

    def __call__(
        self,
        tractogram: Tractogram,
    ):
        """
        Parameters
        ----------
        tractogram: Tractogram
            Tractogram in world space

        Returns
        -------
        filtered_streamlines: Tractogram
            Filtered tractogram according to the oracle in diff world space.
        """

        valid = [len(s) > self.min_nb_steps for s in tractogram.streamlines]
        return tractogram[valid]


class OracleFilter:
    """ TODO
    """

    def __init__(
        self,
        checkpoint: str,
        min_nb_steps: int,
        reference: str,
        affine_vox2rasmm: np.ndarray,
        device: str
    ):
        self.name = 'oracle_filter'

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
        tractogram: Tractogram,
    ):
        """
        Parameters
        ----------
        tractogram: Tractogram
            Tractogram in world space

        Returns
        -------
        filtered_streamlines: Tractogram
            Filtered tractogram according to the oracle in diff world space.
        """
        # Bring tractogram to anat space
        sft = StatefulTractogram(
            streamlines=tractogram.streamlines.copy(),
            reference=self.reference,
            space=Space.RASMM)

        if not self.checkpoint:
            return tractogram

        sft.to_vox()
        sft.to_corner()

        streamlines = sft.streamlines
        if len(streamlines) == 0:
            return tractogram

        batch_size = 4096
        N = len(streamlines)
        predictions = np.zeros((N))
        for i in range(0, N, batch_size):

            j = i + batch_size
            scores = self.model.predict(streamlines[i:j])
            predictions[i:j] = scores

        viable = (predictions > 0.5).astype(bool)

        return tractogram[viable]
