import numpy as np
import torch

from dipy.tracking.streamline import set_number_of_points
from nibabel.streamlines.array_sequence import ArraySequence
from torch import nn

from TrackToLearn.environments.reward import Reward


def format_widths(widths_str):
    return [int(i) for i in widths_str.split('-')]


def make_fc_network(
    widths, input_size, output_size, activation=nn.ReLU, dropout=0.5,
    last_activation=nn.Identity
):
    layers = [nn.Flatten(), nn.Linear(input_size, widths[0]),
              activation(), nn.Dropout(dropout)]
    for i in range(len(widths[:-1])):
        layers.extend(
            [nn.Linear(widths[i], widths[i+1]), activation(),
             nn.Dropout(dropout)])

    layers.extend(
        [nn.Linear(widths[-1], output_size), last_activation()])
    return nn.Sequential(*layers)


class FeedForwardOracle(nn.Module):

    def __init__(self, input_size, output_size, layers, lr):
        super(FeedForwardOracle, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.layers = format_widths(layers)
        self.lr = lr

        self.network = make_fc_network(
            self.layers, self.input_size, self.output_size)

    def forward(self, x):
        return self.network(x).squeeze(-1)

    @classmethod
    def load_from_checkpoint(cls, checkpoint: dict):

        hyper_parameters = checkpoint["hyper_parameters"]

        input_size = hyper_parameters['input_size']
        output_size = hyper_parameters['output_size']
        layers = hyper_parameters['layers']
        lr = hyper_parameters['lr']

        model = FeedForwardOracle(input_size, output_size, layers, lr)

        model_weights = checkpoint["state_dict"]

        # update keys by dropping `auto_encoder.`
        for key in list(model_weights):
            model_weights[key] = \
                model_weights.pop(key)

        model.load_state_dict(model_weights)
        model.eval()

        return model


class AutoencoderOracle(nn.Module):

    def __init__(self, input_size, output_size, layers, lr):
        super(AutoencoderOracle, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = 3
        self.layers = format_widths(layers)
        self.lr = lr

        # TODO: Make the autoencoder architecture parametrizable ?

        self.encoder = nn.Sequential(
            nn.Conv1d(3, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(256, 512, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 3, stride=1, padding=0))

        self.network = make_fc_network(
            self.layers, 1024, self.output_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 3, 3, stride=2, padding=0),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        z = self.encoder(x).squeeze(-1)
        return self.network(z).squeeze(-1)

    @classmethod
    def load_from_checkpoint(cls, checkpoint: dict):

        hyper_parameters = checkpoint["hyper_parameters"]

        input_size = hyper_parameters['input_size']
        output_size = hyper_parameters['output_size']
        layers = hyper_parameters['layers']
        lr = hyper_parameters['lr']

        model = AutoencoderOracle(input_size, output_size, layers, lr)

        model_weights = checkpoint["state_dict"]

        # update keys by dropping `auto_encoder.`
        for key in list(model_weights):
            model_weights[key] = \
                model_weights.pop(key)

        model.load_state_dict(model_weights)
        model.eval()

        return model


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
        if checkpoint:
            self.checkpoint = torch.load(checkpoint)

            hyper_parameters = self.checkpoint["hyper_parameters"]
            # The model's class is saved in hparams
            models = {
                'AutoencoderOracle': AutoencoderOracle,
                'FeedForwardOracle': FeedForwardOracle
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
        if L > 3:

            array_seq = ArraySequence(streamlines)

            resampled_streamlines = set_number_of_points(array_seq, 128)
            # Compute streamline features as the directions between points
            dirs = np.diff(resampled_streamlines, axis=1)

            # Load the features as torch tensors and predict
            with torch.no_grad():
                data = torch.as_tensor(
                    dirs, dtype=torch.float, device=self.device)
                predictions = self.model(data).cpu().numpy()

            return dones.astype(int) * predictions
        return np.zeros((N))
