import numpy as np


def _harvest_states(self, i, *args):
    return (a[:, i, ...] for a in args)


def stack_states(self, full, single):
    if full[0] is not None:
        return (np.vstack((f, s[None, ...]))
                for (f, s) in zip(full, single))
    else:
        return (s[None, :, ...] for s in single)
