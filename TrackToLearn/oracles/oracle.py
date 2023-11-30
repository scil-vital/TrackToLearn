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

    def __init__(self, checkpoint: str, device: str, batch_size=4096):
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
        self.batch_size = batch_size

        self.device = device

    def predict(self, streamlines):
        # Total number of predictions to return
        N = len(streamlines)
        # Placeholders for input and output data
        placeholder = torch.zeros(
            (self.batch_size, 127, 3), pin_memory=True)
        result = torch.zeros((N), dtype=torch.float, device=self.device)

        # Get the first batch
        batch = streamlines[:self.batch_size]
        N_batch = len(batch)
        # Resample streamlines to fixed number of point to set all
        # sequences to same length
        data = set_number_of_points(batch, 128)
        # Compute streamline features as the directions between points
        dirs = np.diff(data, axis=1)
        # Send the directions to pinned memory
        placeholder[:N_batch] = torch.from_numpy(dirs)
        # Send the pinned memory to GPU asynchronously
        input_data = placeholder[:N_batch].to(
            self.device, non_blocking=True, dtype=torch.float)
        i = 0

        while i <= N // self.batch_size:
            start = (i+1) * self.batch_size
            end = min(start + self.batch_size, N)
            # Prefetch the next batch
            if start < end:
                batch = streamlines[start:end]
                # Resample streamlines to fixed number of point to set all
                # sequences to same length
                data = set_number_of_points(batch, 128)
                # Compute streamline features as the directions between points
                dirs = np.diff(data, axis=1)
                # Put the directions in pinned memory
                placeholder[:end-start] = torch.from_numpy(dirs)

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    predictions = self.model(input_data)
                    result[
                        i * self.batch_size:
                        (i * self.batch_size) + self.batch_size] = predictions
            i += 1
            if i >= N // self.batch_size:
                break
            # Send the pinned memory to GPU asynchronously
            input_data = placeholder[:end-start].to(
                self.device, non_blocking=True, dtype=torch.float)

        return result.cpu().numpy()
