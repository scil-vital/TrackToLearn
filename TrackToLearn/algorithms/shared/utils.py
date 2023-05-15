import numpy as np

from torch import nn


def add_item_to_means(means, dic):
    return {k: means[k] + [dic[k]] for k in dic.keys()}


def add_to_means(means, dic):
    return {k: means[k] + dic[k] for k in dic.keys()}


def mean_losses(dic):
    return {k: np.mean(dic[k]) for k in dic.keys()}


def harvest_states(i, *args):
    return (a[:, i, ...] for a in args)


def stack_states(full, single):
    if full[0] is not None:
        return (np.vstack((f, s[None, ...]))
                for (f, s) in zip(full, single))
    else:
        return (s[None, :, ...] for s in single)


def format_widths(widths_str):
    return [int(i) for i in widths_str.split('-')]


def make_fc_network(
    widths, input_size, output_size, activation=nn.ReLU,
    last_activation=nn.Identity
):
    layers = [nn.Linear(input_size, widths[0]), activation()]
    for i in range(len(widths[:-1])):
        layers.extend(
            [nn.Linear(widths[i], widths[i+1]), activation()])
    # no activ. on last layer
    layers.extend([nn.Linear(widths[-1], output_size)])
    return nn.Sequential(*layers)
