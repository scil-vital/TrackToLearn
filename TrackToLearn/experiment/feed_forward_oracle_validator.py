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
