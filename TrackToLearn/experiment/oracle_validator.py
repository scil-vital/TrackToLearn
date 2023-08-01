import numpy as np
import torch

from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import set_number_of_points

from TrackToLearn.experiment.autoencoder_oracle_validator import (
    AutoencoderOracle)
from TrackToLearn.experiment.feed_forward_oracle_validator import (
    FeedForwardOracle)
from TrackToLearn.experiment.transformer_oracle_validator import (
    TransformerOracle)
from TrackToLearn.experiment.validators import Validator


class OracleValidator(Validator):

    def __init__(self, checkpoint, reference, device):

        self.name = 'Oracle'

        self.checkpoint = torch.load(checkpoint)
        self.reference = reference

        hyper_parameters = self.checkpoint["hyper_parameters"]
        # The model's class is saved in hparams
        models = {
            'AutoencoderOracle': AutoencoderOracle,
            'FeedForwardOracle': FeedForwardOracle,
            'TransformerOracle': TransformerOracle
        }

        # Load it from the checkpoint
        self.model = models[hyper_parameters[
            'name']].load_from_checkpoint(self.checkpoint).to(device)
        self.device = device

    def __call__(self, filename):

        sft = load_tractogram(filename, self.reference,
                              bbox_valid_check=False, trk_header_check=False)

        sft.to_vox()
        sft.to_corner()

        streamlines = sft.streamlines
        # Resample streamlines to a fixed number of points. This should be
        # set by the model ? TODO?

        resampled_streamlines = set_number_of_points(streamlines, 128)

        # Compute streamline features as the directions between points
        dirs = np.diff(resampled_streamlines, axis=1)

        batch_size = 4096
        predictions = []
        for i in range(0, len(dirs), batch_size):
            j = i + batch_size
            # Load the features as torch tensors and predict
            with torch.no_grad():
                data = torch.as_tensor(
                    dirs[i:j], dtype=torch.float, device='cuda')
                pred_batch = self.model(data).cpu().numpy().tolist()
                predictions.extend(pred_batch)

        predictions = np.asarray(predictions)

        return {'Oracle': np.mean(predictions > 0.5)}
