import numpy as np
import torch

from dipy.tracking.streamline import set_number_of_points
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Tractogram
from fury import window, actor

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
        reference: str,
        diff_affine: np.ndarray,
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

        self.model.eval()
        self.diff_affine = diff_affine
        self.reference = reference
        self.min_nb_steps = min_nb_steps
        self.device = device

    # def __call__(
    #     self,
    #     streamlines: np.ndarray,
    #     dones: np.ndarray
    # ):
    #     """
    #     Parameters
    #     ----------
    #     streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
    #         Streamline coordinates in voxel space

    #     Returns
    #     -------
    #     rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
    #         Array containing the reward
    #     """

    #     # Resample streamlines to a fixed number of points. This should be
    #     # set by the model ? TODO?
    #     N, L, P = streamlines.shape
    #     if L > 3:

    #         # TODO: What the actual fuck
    #         tractogram = Tractogram(
    #             streamlines=streamlines.copy())
    #         tractogram.apply_affine(self.diff_affine)

    #         sft = StatefulTractogram(
    #             streamlines=tractogram.streamlines,
    #             reference=self.reference,
    #             space=Space.RASMM)

    #         sft.to_vox()
    #         sft.to_corner()

    #         resampled_streamlines = set_number_of_points(sft.streamlines, 128)
    #         # Compute streamline features as the directions between points
    #         dirs = np.diff(resampled_streamlines, axis=1)

    #         # Load the features as torch tensors and predict
    #         with torch.no_grad():
    #             with torch.cuda.amp.autocast():
    #                 data = torch.as_tensor(
    #                     dirs, dtype=torch.float, device=self.device)
    #                 predictions = self.model(data)

    #         return predictions.cpu().numpy()

    #     return np.zeros((N))

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
        if L > 3:

            # TODO: What the actual fuck
            tractogram = Tractogram(
                streamlines=streamlines)
            tractogram.apply_affine(self.diff_affine)

            sft = StatefulTractogram(
                streamlines=tractogram.streamlines,
                reference=self.reference,
                space=Space.RASMM)

            sft.to_vox()
            sft.to_corner()

            resampled_streamlines = set_number_of_points(sft.streamlines, 128)
            # Compute streamline features as the directions between points
            dirs = np.diff(resampled_streamlines, axis=1)

            # Load the features as torch tensors and predict
            with torch.cuda.amp.autocast():
                data = torch.as_tensor(
                    dirs, dtype=torch.float, device=self.device)
                predictions = self.model(data)

            scores = torch.zeros_like(predictions, device=self.device)
            scores[predictions > 0.5] = 1.
            return scores.cpu().numpy() * dones.astype(int)

        return np.zeros((N))

    def render(self, streamlines, predictions=None):

        scene = window.Scene()

        line_actor = actor.streamtube(
            streamlines, linewidth=1.0, colors=predictions)
        scene.add(line_actor)

        showm = window.ShowManager(scene, reset_camera=True)
        showm.initialize()
        showm.start()
