import numpy as np


def harvest_states(i, *args):
    return (a[:, i, ...] for a in args)


def stack_states(full, single):
    if full[0] is not None:
        return (np.vstack((f, s[None, ...]))
                for (f, s) in zip(full, single))
    else:
        return (s[None, :, ...] for s in single)
