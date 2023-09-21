from torch import nn


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
