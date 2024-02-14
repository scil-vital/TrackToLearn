import nibabel as nib
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
        min_nb_steps: int,
        reference: nib.Nifti1Image,
        affine_vox2rasmm: np.ndarray,
        device: str
    ):
        # Name for stats
        self.name = 'oracle_reward'
        # If the reward is dense or not
        self.dense = dense
        # Minimum number of steps before giving reward
        # Only useful for 'sparse' reward
        self.min_nb_steps = min_nb_steps
        # Checkpoint of the oracle, which contains weights and hyperparams.
        if checkpoint:
            self.checkpoint = checkpoint
            # The oracle is declared as a singleton to prevent loading the
            # weights in memory multiple times.
            self.model = OracleSingleton(checkpoint, device)
        else:
            self.checkpoint = None

        self.reference = reference
        self.affine_vox2rasmm = affine_vox2rasmm

        # Reference anat
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

        # In the "dense" version, the reward per point is simply
        # the direct score.
        scores = self.model.predict(streamlines)

        return scores

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
        N = dones.shape[0]
        reward = np.zeros((N))
        predictions = self.model.predict(streamlines)
        # Double indexing to get the indexes. Don't forget you
        # can't assign using double indexes as the first indexing
        # will return a copy of the array.
        idx = np.arange(N)[dones][predictions > 0.5]
        # Assign the reward using the precomputed double indexes.
        reward[idx] = 1.0
        return reward

    def __call__(
        self,
        streamlines: np.ndarray,
        dones: np.ndarray,
    ):

        N, L, P = streamlines.shape
        if L > self.min_nb_steps and sum(dones.astype(int)) > 0:

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

            if self.dense:
                return self.dense_reward(sft.streamlines, dones)
            return self.sparse_reward(sft.streamlines, dones)
        return np.zeros((N))

    def render(self, streamlines, predictions=None):

        scene = window.Scene()

        line_actor = actor.streamtube(
            streamlines, linewidth=1.0, colors=predictions)
        scene.add(line_actor)

        showm = window.ShowManager(scene, reset_camera=True)
        showm.initialize()
        showm.start()
