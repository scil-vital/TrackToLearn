import numpy as np


class Reward(object):

    """ Abstract function that all "rewards" must implement.
    """

    def __call__(
        streamlines: np.ndarray,
        dones: np.ndarray
    ):
        assert False, "Not implemented"


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

    def __call__(self, streamlines, dones):
        """
        Each reward component is weighted according to a
        coefficient and then summed.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        dones: `numpy.ndarray` of shape (n_streamlines)
            Whether tracking is over for each streamline or not.

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
                rewards_factors[i] = w * f(streamlines, dones)

        reward = np.sum(rewards_factors, axis=0)

        return reward
