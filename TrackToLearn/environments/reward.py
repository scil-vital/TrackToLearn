import numpy as np


class Reward(object):

    """ Abstract function that all "rewards" must implement.
    """

    def __call__(
        self,
        streamlines: np.ndarray,
        bundle_idx: np.ndarray,
        flags: np.ndarray
    ):
        self.name = "Undefined"

        assert False, "Not implemented"

    def reset(self):
        """ Most reward factors do not need to be reset.
        """
        pass


class RewardFunction():

    """ Compute the reward function as the sum of its weighted factors.
    Each factor may reward streamlines "densely" (i.e. at every step) or
    "sparsely" (i.e. once per streamline).

    """

    def __init__(
        self,
        factors,
        weights,
    ):
        """
        """
        assert len(factors) == len(weights)

        self.factors = factors
        self.weights = weights

        self.F = len(self.factors)

    def __call__(self, streamlines, bundles_idx, flags):
        """
        Each reward component is weighted according to a
        coefficient and then summed.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        bundles_idx: `numpy.ndarray` of shape (n_streamlines)
            Bundle corresponding to every streamline
        flags: `numpy.ndarray` of shape (n_streamlines)
            Flags for stopping criterion.

        Returns
        -------
        rewards: np.ndarray of floats
            Reward components weighted by their coefficients as well
            as the penalites
        """

        N = len(streamlines)

        rewards_factors = np.zeros((self.F, N))

        for i, (w, f) in enumerate(zip(self.weights, self.factors)):
            if w > 0:
                rewards_factors[i] = w * f(streamlines, bundles_idx, flags)

        info = {}
        for i, f in enumerate(self.factors):
            info[f.name] = np.mean(rewards_factors[i])

        reward = np.sum(rewards_factors, axis=0)

        return reward, info

    def reset(self):
        """
        """

        for f in self.factors:
            f.reset()
