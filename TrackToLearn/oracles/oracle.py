import numpy as np
import torch
from dipy.tracking.streamline import set_number_of_points

from TrackToLearn.oracles.autoencoder_oracle import AutoencoderOracle
from TrackToLearn.oracles.feed_forward_oracle import FeedForwardOracle
from TrackToLearn.oracles.transformer_oracle import TransformerOracle


class OracleSingleton:
    _self = None

    def __new__(cls, *args, **kwargs):
        if cls._self is None:
            print('Instanciating new Oracle, should only happen once.')
            cls._self = super().__new__(cls)
        return cls._self

    def __init__(self, checkpoint: str, device: str):
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
        self.device = device

    def predict(self, streamlines):
        # Resample streamlines to fixed number of point to set all
        # sequences to same length
        resampled_streamlines = set_number_of_points(streamlines, 128)
        # Compute streamline features as the directions between points
        dirs = np.diff(resampled_streamlines, axis=1)

        # Load the features as torch tensors and predict
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                data = torch.as_tensor(
                    dirs, dtype=torch.float, device=self.device)
                predictions = self.model(data)
                scores = predictions.detach().cpu().numpy()

        return scores
