import numpy as np
from torch import nn


def _harvest_states(self, i, *args):
    return (a[:, i, ...] for a in args)


def stack_states(self, full, single):
    if full[0] is not None:
        return (np.vstack((f, s[None, ...]))
                for (f, s) in zip(full, single))
    else:
        return (s[None, :, ...] for s in single)


def format_widths(widths_str):
    return [int(i) for i in widths_str.split('-')]


def make_fc_network(
    widths, input_size, output_size, activation=nn.ReLU,
    last_activation=nn.Identity, dropout=0.0
):
    layers = [nn.Linear(input_size, widths[0]), activation()]
    for i in range(len(widths[:-1])):
        layers.extend(
            [nn.Linear(widths[i], widths[i+1]), activation(),
             nn.Dropout(dropout)])
    # no activ. on last layer
    layers.extend([nn.Linear(widths[-1], output_size), last_activation()])
    return nn.Sequential(*layers)


def make_rnn_network(
    widths, input_size, output_size, n_recurrent, activation=nn.ReLU,
    last_activation=nn.Identity, dropout=0.0
):
    rnn = nn.LSTM(input_size, widths[0], num_layers=n_recurrent,
                  dropout=dropout, batch_first=True)
    layers = [activation()]
    for i in range(len(widths[:-1])):
        layers.extend(
            [nn.Linear(widths[i], widths[i+1]), activation(),
             nn.Dropout(dropout)])
    # no activ. on last layer
    layers.extend([nn.Linear(widths[-1], output_size), last_activation()])

    # no activ. on last layer
    decoder = nn.Sequential(*layers)
    return rnn, decoder
