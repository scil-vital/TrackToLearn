import numpy as np
import torch

from dipy.tracking.streamline import set_number_of_points
from fury import window, actor
from nibabel.streamlines.array_sequence import ArraySequence

from TrackToLearn.environments.reward import Reward

from TrackToLearn.experiment.autoencoder_oracle_validator import (
    AutoencoderOracle)
from TrackToLearn.experiment.feed_forward_oracle_validator import (
    FeedForwardOracle)
from TrackToLearn.experiment.transformer_oracle_validator import (
    TransformerOracle)


class OracleReward(Reward):

    """ Reward streamlines based on their alignment with local peaks
    and their past direction.
    """

    def __init__(
        self,
        checkpoint: str,
        min_nb_steps: int,
        device: str
    ):
        self.name = 'oracle_reward'

        if checkpoint:
            self.checkpoint = torch.load(checkpoint)

            hyper_parameters = self.checkpoint["hyper_parameters"]
            # The model's class is saved in hparams
            # The model's class is saved in hparams
            models = {
                'AutoencoderOracle': AutoencoderOracle,
                'FeedForwardOracle': FeedForwardOracle,
                'TransformerOracle': TransformerOracle
            }

            # Load it from the checkpoint
            self.model = models[hyper_parameters[
                'name']].load_from_checkpoint(self.checkpoint).to(device)

        self.min_nb_steps = min_nb_steps
        self.device = device

    def __call__(
        self,
        streamlines: np.ndarray,
        dones: np.ndarray
    ):
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

        # Resample streamlines to a fixed number of points. This should be
        # set by the model ? TODO?
        N, L, P = streamlines.shape
        # Ensure that streamlines are long enough to be resampled
        # and that atleast one streamline is 'done' being tracked.
        if L > 3 and sum(dones):

            array_seq = ArraySequence(streamlines)

            resampled_streamlines = set_number_of_points(array_seq, 128)
            # Compute streamline features as the directions between points
            # Only compute reward for streamlines that are 'done'
            # to save resources.
            dirs = np.diff(resampled_streamlines[dones], axis=1)

            # Load the features as torch tensors and predict
            with torch.no_grad():
                data = torch.as_tensor(
                    dirs, dtype=torch.float, device=self.device)
                predictions = self.model(data).cpu().numpy()

            scores = np.copy(dones).astype(int)
            scores[dones] = predictions > 0.5
            return scores

            # return scores * dones.astype(int)

        return np.zeros((N))

    def render(self, streamlines, predictions=None):

        scene = window.Scene()

        line_actor = actor.streamtube(
            streamlines, linewidth=1.0)
        scene.add(line_actor)

        showm = window.ShowManager(scene, reset_camera=True)
        showm.initialize()
        showm.start()
