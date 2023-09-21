import numpy as np

from dipy.io.stateful_tractogram import StatefulTractogram, Space, Tractogram
from fury import window, actor

from TrackToLearn.environments.reward import Reward

from TrackToLearn.oracles.oracle import OracleSingleton


class OracleReward(Reward):

    """ Reward streamlines based on their alignment with local peaks
    and their past direction.
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

    def dense_reward(self, streamlines, dones):
        """
        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space

        Returns
        -------
        rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
            Array containing the reward
        """
        if not self.checkpoint:
            return None

        # Resample streamlines to a fixed number of points. This should be
        # set by the model ? TODO?
        N, L, P = streamlines.shape
        if L > 3:

            rasmm_streamlines = streamlines * self.affine_vox2rasmm[0][0]

            # TODO: What the actual fuck
            tractogram = Tractogram(
                streamlines=rasmm_streamlines)

            sft = StatefulTractogram(
                streamlines=tractogram.streamlines,
                reference=self.reference,
                space=Space.RASMM)

            sft.to_vox()
            sft.to_corner()

            scores = self.model.predict(sft.streamlines)

            return scores

        return np.zeros((N))

    def sparse_reward(self, streamlines, dones):
        """
        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space

        Returns
        -------
        rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
            Array containing the reward
        """
        if not self.checkpoint:
            return None

        # Resample streamlines to a fixed number of points. This should be
        # set by the model ? TODO?
        N, L, P = streamlines.shape
        if L > 3:

            # rasmm_streamlines = streamlines * self.affine_vox2rasmm[0][0]

            # TODO: What the actual fuck
            tractogram = Tractogram(
                streamlines=streamlines)
            tractogram.apply_affine(self.affine_vox2rasmm)

            sft = StatefulTractogram(
                streamlines=tractogram.streamlines,
                reference=self.reference,
                space=Space.RASMM)

            sft.to_vox()
            sft.to_corner()

            scores = self.model.predict(sft.streamlines)
            reward = np.zeros_like(scores)
            reward[scores > 0.5] = 1.0
            return reward * dones.astype(int)

        return np.zeros((N))

    def __call__(
        self,
        streamlines: np.ndarray,
        dones: np.ndarray
    ):
        return self.dense_reward(streamlines, dones)

    def render(self, streamlines, predictions=None):

        scene = window.Scene()

        line_actor = actor.streamtube(
            streamlines, linewidth=1.0, colors=predictions)
        scene.add(line_actor)

        showm = window.ShowManager(scene, reset_camera=True)
        showm.initialize()
        showm.start()
