import numpy as np

from dipy.io.stateful_tractogram import StatefulTractogram
from nibabel.streamlines import Tractogram

from TrackToLearn.environments.tracking_env import TrackingEnvironment
from TrackToLearn.environments.stopping_criteria import (
    is_flag_set,
    StoppingFlags)


class BackwardTrackingEnvironment(TrackingEnvironment):
    """ Pre-initialized environment. Tracking will start at the seed from
    flipped half-streamlines.
    """

    def reset(self, streamlines: np.ndarray) -> np.ndarray:
        """ Initialize tracking based on half-streamlines.

        Parameters
        ----------
        streamlines : list
            Half-streamlines to initialize environment

        Returns
        -------
        state: numpy.ndarray
            Initial state for RL model
        """

        # Half-streamlines
        self.seeding_streamlines = [s[:] for s in streamlines]
        N = len(streamlines)

        # Jagged arrays ugh
        # This is dirty, clean up asap
        self.half_lengths = np.asarray(
            [len(s) for s in self.seeding_streamlines])
        max_half_len = max(self.half_lengths)
        half_streamlines = np.zeros(
            (N, max_half_len, 3), dtype=np.float32)

        for i, s in enumerate(self.seeding_streamlines):
            le = self.half_lengths[i]
            half_streamlines[i, :le, :] = s

        self.initial_points = np.asarray([s[0] for s in streamlines])

        # Initialize seeds as streamlines
        self.streamlines = np.concatenate((np.zeros(
            (N, self.max_nb_steps + 1, 3),
            dtype=np.float32), half_streamlines), axis=1)

        self.streamlines = np.flip(self.streamlines, axis=1)
        # This means that all streamlines in the batch are limited by the
        # longest half-streamline :(
        self.lengths = np.ones(N, dtype=np.int32) * max_half_len

        # Done flags for tracking backwards
        self.dones = np.full(N, False)
        self.max_half_len = max_half_len
        self.length = max_half_len
        self.continue_idx = np.arange(N)
        self.flags = np.zeros(N, dtype=int)

        # Signal
        return self._format_state(self.streamlines[:, :self.length])

    def get_streamlines(self) -> StatefulTractogram:

        tractogram = Tractogram()
        # Get both parts of the streamlines.
        stopped_streamlines = [self.streamlines[
            i, self.max_half_len - self.half_lengths[i]:self.lengths[i], :]
            for i in range(len(self.streamlines))]

        # Remove last point if the resulting segment had an angle too high.
        flags = is_flag_set(
            self.flags, StoppingFlags.STOPPING_CURVATURE)
        stopped_streamlines = [
            s[:-1] if f else s for f, s in zip(flags, stopped_streamlines)]

        stopped_seeds = self.initial_points

        # Harvested tractogram
        tractogram = Tractogram(
            streamlines=stopped_streamlines,
            data_per_streamline={"seeds": stopped_seeds,
                                 },
            affine_to_rasmm=self.affine_vox2rasmm)

        return tractogram
