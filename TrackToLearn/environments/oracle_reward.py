import numpy as np

from dipy.io.stateful_tractogram import StatefulTractogram, Space, Tractogram
from fury import window, actor

from TrackToLearn.environments.reward import Reward

from TrackToLearn.oracles.oracle import OracleSingleton


class OracleReward(Reward):

    """ Reward streamlines based on the predicted scores of an "Oracle".
    In the "dense" version, a reward if given at each point by the oracle.
    In the "sparse" version, a binary reward is given by the oracle at the
    end of tracking.
    """

    def __init__(
        self,
        checkpoint: str,
        dense: bool,
        reference: str,
        affine_vox2rasmm: np.ndarray,
        device: str
    ):
        # Name for stats
        self.name = 'oracle_reward'
        # If the reward is dense or not
        self.dense = dense
        # Checkpoint of the oracle, which contains weights and hyperparams.
        if checkpoint:
            self.checkpoint = checkpoint
            # The oracle is declared as a singleton to prevent loading the
            # weights in memory multiple times.
            self.model = OracleSingleton(checkpoint, device)
        else:
            self.checkpoint = None

        # Affine from vox to world diff space.
        self.affine_vox2rasmm = affine_vox2rasmm
        # Reference anat
        self.reference = reference
        # Device to load the oracle.
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

            # Change ref of streamlines. This is weird on the ISMRM2015
            # dataset as the diff and anat are not in the same space,
            # but it should be fine on other datasets.
            tractogram = Tractogram(
                streamlines=streamlines.copy())
            # It is unclear whether I should apply the affine or just
            # multiply by the voxel size.
            tractogram.apply_affine(self.affine_vox2rasmm)
            sft = StatefulTractogram(
                streamlines=tractogram.streamlines,
                reference=self.reference,
                space=Space.RASMM)
            sft.to_vox()
            sft.to_corner()

            # In the "dense" version, the reward per point is simply
            # the direct score.
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
        if L > 3 and sum(dones.astype(int)) > 0:

            # Change ref of streamlines. This is weird on the ISMRM2015
            # dataset as the diff and anat are not in the same space,
            # but it should be fine on other datasets.
            tractogram = Tractogram(
                streamlines=streamlines.copy()[dones])
            tractogram.apply_affine(self.affine_vox2rasmm)
            sft = StatefulTractogram(
                streamlines=tractogram.streamlines,
                reference=self.reference,
                space=Space.RASMM)
            sft.to_vox()
            sft.to_corner()

            # Get scores from the oracle
            scores = self.model.predict(sft.streamlines)
            reward = np.zeros((N))
            # Double indexing to get the indexes. Don't forget you
            # can't assign using double indexes as the first indexing
            # will return a copy of the array.
            idx = np.arange(N)[dones][scores > 0.5]
            # Assign the reward using the precomputed double indexes.
            reward[idx] = 1.0
            return reward

        return np.zeros((N))

    def __call__(
        self,
        streamlines: np.ndarray,
        dones: np.ndarray
    ):
        if self.dense:
            return self.dense_reward(streamlines, dones)
        return self.sparse_reward(streamlines, dones)

    def render(self, streamlines, predictions=None):

        scene = window.Scene()

        line_actor = actor.streamtube(
            streamlines, linewidth=1.0, colors=predictions)
        scene.add(line_actor)

        showm = window.ShowManager(scene, reset_camera=True)
        showm.initialize()
        showm.start()
