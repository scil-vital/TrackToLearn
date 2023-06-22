import numpy as np
import torch

from challenge_scoring.metrics.scoring import score_submission
from challenge_scoring.utils.attributes import load_attribs

from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import set_number_of_points
from torch import nn
from os.path import join as pjoin


class Validator(object):

    def __init__(self):

        self.name = ''

    def __call__(self, filename):

        assert False, 'not implemented'


class TractometerValidator(Validator):

    def __init__(self, base_dir):

        self.name = 'Tractometer'
        self.base_dir = base_dir

    def __call__(self, filename):

        #  Load bundle attributes for tractometer
        # TODO: No need to load this every time, should only be loaded
        # once
        gt_bundles_attribs_path = pjoin(
            self.base_dir, 'gt_bundles_attributes.json')
        basic_bundles_attribs = load_attribs(gt_bundles_attribs_path)

        # Score tractogram
        scores = score_submission(
            filename,
            self.base_dir,
            basic_bundles_attribs,
            compute_ic_ib=True)
        cleaned_scores = {}
        for k, v in scores.items():
            if type(v) in [float, int]:
                cleaned_scores.update({k: v})
        return cleaned_scores


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


class OracleValidator(Validator):

    def __init__(self, checkpoint, reference, device):

        self.name = 'Oracle'

        self.checkpoint = torch.load(checkpoint)
        self.reference = reference

        hyper_parameters = self.checkpoint["hyper_parameters"]
        # The model's class is saved in hparams
        models = {
            'AutoencoderOracle': AutoencoderOracle,
            'FeedForwardOracle': FeedForwardOracle
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

        # Load the features as torch tensors and predict
        with torch.no_grad():
            data = torch.as_tensor(
                dirs, dtype=torch.float, device=self.device)
            predictions = self.model(data).cpu().numpy()

        return {'Oracle': np.mean(predictions > 0.5)}
