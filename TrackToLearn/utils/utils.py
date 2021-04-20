import os
import sys
import timeit

from os.path import join as pjoin
from time import time

from collections import deque

import numpy as np

COLOR_CODES = {
    'black': '\u001b[30m',
    'red': '\u001b[31m',
    'green': '\u001b[32m',
    'yellow': '\u001b[33m',
    'blue': '\u001b[34m',
    'magenta': '\u001b[35m',
    'cyan': '\u001b[36m',
    'white': '\u001b[37m',
    'reset': '\u001b[0m'
}


class ExplorationScheduler():

    def __init__(self, n_epochs, start, end):
        self.start = start
        self.end = end
        self.n_epochs = n_epochs

        self.schedule = np.geomspace(
            start, end, num=self.n_epochs, endpoint=True)[::-1]
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        val = self.schedule[self.index]
        if self.index < len(self.schedule):
            self.index += 1
        return val

    @property
    def first(self):
        return self.schedule[0]

    @property
    def value(self):
        return self.schedule[self.index]


class KappaScheduler(ExplorationScheduler):

    def __init__(self, n_epochs, start, end):
        super().__init__(n_epochs, start, end)
        self.schedule = np.geomspace(
            start, end, num=self.n_epochs, endpoint=True)


class LossHistory(object):
    """ History of the loss during training.
    Usage:
        monitor = LossHistory()
        ...
        # Call update at each iteration
        monitor.update(2.3)
        ...
        monitor.avg  # returns the average loss
        ...
        monitor.end_epoch()  # call at epoch end
        ...
        monitor.epochs  # returns the loss curve as a list
    """

    def __init__(self, name, filename, path):
        self.name = name
        self.history = []
        self.epochs = []
        self.sum = 0.0
        self.count = 0
        self._avg = 0.0
        self.num_iter = 0
        self.num_epochs = 0

        self.filename = filename
        self.path = path

    def __len__(self):
        return len(self.history)

    def update(self, value):
        if np.isinf(value):
            return

        self.history.append(value)
        self.sum += value
        self.count += 1
        self._avg = self.sum / self.count
        self.num_iter += 1

    @property
    def avg(self):
        return self._avg

    def end_epoch(self, epoch):
        self.epochs.append((epoch, self._avg))
        self.sum = 0.0
        self.count = 0
        self._avg = 0.0
        self.num_epochs += 1

        directory = os.path.dirname(pjoin(self.path, 'plots'))
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(pjoin(directory, '{}.npy'.format(self.filename)), 'wb') as f:
            np.save(f, self.epochs)


class IterTimer(object):
    def __init__(self, history_len=5):
        self.history = deque(maxlen=history_len)
        self.iterable = None
        self.start_time = None

    def __call__(self, iterable):
        self.iterable = iter(iterable)
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self.start_time is not None:
            elapsed = timeit.default_timer() - self.start_time
            self.history.append(elapsed)
        self.start_time = timeit.default_timer()
        return next(self.iterable)

    @property
    def mean(self):
        return np.mean(self.history) if len(self.history) > 0 else 0


class Timer:
    """ Times code within a `with` statement, optionally adding color. """

    def __init__(self, txt, newline=False, color=None):
        try:
            prepend = (COLOR_CODES[color] if color else '')
            append = (COLOR_CODES['reset'] if color else '')
        except KeyError:
            prepend = ''
            append = ''

        self.txt = prepend + txt + append
        self.newline = newline

    def __enter__(self):
        self.start = time()
        if not self.newline:
            print(self.txt + "... ", end="")
            sys.stdout.flush()
        else:
            print(self.txt + "... ")

    def __exit__(self, type, value, tb):
        if self.newline:
            print(self.txt + " done in ", end="")

        print("{:.2f} sec.".format(time() - self.start))


'''
Pick a point uniformly from the unit circle
'''


def circle_uniform_pick(size, out=None):
    if out is None:
        out = np.empty((size, 2))

    angle = 2 * np.pi * np.random.random(size)
    out[:, 0], out[:, 1] = np.cos(angle), np.sin(angle)

    return out


def cross_product_matrix(U):
    return np.array([[0., -U[2],  U[1]],
                     [U[2],    0., -U[0]],
                     [-U[1],  U[0],    0.]])


class SingularityError(Exception):
    def __init__(self):
        pass


'''
Von Mises-Fisher distribution, ie. isotropic Gaussian distribution defined over
a sphere.
  mu  =>   mean direction
  kappa => concentration
Uses numerical tricks described in "Numerically stable sampling of the von
Mises Fisher distribution on S2 (and other tricks)" by Wenzel Jakob
Uses maximum likelyhood estimators described in "Modeling Data using
Directional Distributions" by Inderjit S. Dhillon and Suvrit Sra
'''


class VonMisesFisher3(object):
    def __init__(self, mu, kappa):
        self.mu = mu
        self.kappa = kappa

        self.pdf_constant = self.kappa / \
            ((2 * np.pi) * (1. - np.exp(-2. * self.kappa)))
        self.log_pdf_constant = np.log(self.pdf_constant)

    '''
    Generates samples from the distribution
    '''

    def sample(self, size, out=None):
        # Generate the samples for mu=(0, 0, 1)
        eta = np.random.random(size)
        tmp = 1. - (((eta - 1.) / eta) * np.exp(-2. * self.kappa))
        W = 1. + (np.log(eta) + np.log(tmp)) / self.kappa

        V = np.empty((size, 2))
        circle_uniform_pick(size, out=V)
        V *= np.sqrt(1. - W ** 2)[:, None]

        if out is None:
            out = np.empty((size, 3))

        out[:, 0], out[:, 1], out[:, 2] = V[:, 0], V[:, 1], W

        # Rotate the samples to the distribution's mu
        angle = np.arccos(self.mu[2])
        if not np.allclose(angle, .0):
            axis = np.array((-self.mu[1], -self.mu[0], 0.))
            axis /= np.sqrt(np.sum(axis ** 2))
            rot = np.cos(angle) * np.identity(3) + np.sin(angle) * \
                cross_product_matrix(axis) + (1. - np.cos(angle)
                                              ) * np.outer(axis, axis)
            out = np.dot(out, rot)

        # Job done
        return out

    '''
    Returns the probability for X to be generated by the distribution
    '''

    def pdf(self, X):
        if self.kappa == 0.:
            return .25 / np.pi
        else:
            return self.pdf_constant * \
                    np.exp(self.kappa *
                           (self.mu.dot(X) - 1.))

    '''
        Returns the log-probability for X to be generated by the distribution
        '''

    def log_pdf(self, X):
        if self.kappa == 0.:
            return np.log(.25 / np.pi)
        else:
            return self.log_pdf_constant + self.kappa * (self.mu.dot(X) - 1.)

    def __repr__(self):
        return 'VonMisesFisher3(mu = %s, kappa = %f)' % \
                (repr(self.mu), self.kappa)

    @staticmethod
    def _get_kappa(R_bar):
        f = 1. - R_bar ** 2
        if np.allclose(f, 0.):
            raise SingularityError()
        return (R_bar * (3. - R_bar ** 2)) / f

    '''
        Returns an approximation of the most likely VMF to generate samples
          - Assumes that the samples lies on the unit sphere
        '''
    @staticmethod
    def estimate(samples):
        X = np.asarray(samples)

        # Estimate for mu
        S = np.sum(X, axis=0)
        norm_S = np.sqrt(np.sum(S ** 2))
        mu = S / norm_S

        # Initial estimate for kappa
        R_bar = norm_S / X.shape[0]
        kappa = VonMisesFisher3._get_kappa(R_bar)

        # Refine kappa estimate
        # TODO

        # Job done
        return VonMisesFisher3(mu, kappa)

    '''
        Returns an approximation of the most likely VMF to generate samples.
        A weight vector specify the relative signifiance of each sample.
          - Assumes that the samples lies on the unit sphere
          - Assumes that the weight vector sum equals 1.
        '''
    @staticmethod
    def estimate_weighted(samples, log_weights):
        X = np.asarray(samples)

        # Estimate for mu
        S = np.sum(X * log_weights[:, None], axis=0)
        norm_S = np.sqrt(np.sum(S ** 2))
        mu = S / norm_S

        # Initial estimate for kappa
        R_bar = norm_S
        kappa = VonMisesFisher3._get_kappa(R_bar)

        # Refine kappa estimate
        # TODO

        # Job done
        return VonMisesFisher3(mu, kappa)
