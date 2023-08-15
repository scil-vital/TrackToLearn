# import numpy as np
import torch


class Reward(object):

    """ Abstract function that all "rewards" must implement.
    """

    def __call__(
        self,
        streamlines: torch.tensor,
        dones: torch.tensor
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
        rewards: torch.tensor of floats
            Reward components weighted by their coefficients as well
            as the penalites
        """

        N = len(streamlines)

        rewards_factors = torch.zeros((self.F, N), device=streamlines.device)

        for i, (w, f) in enumerate(zip(self.weights, self.factors)):
            if w > 0:
                rewards_factors[i] = w * f(streamlines, dones)

        info = {}
        for i, f in enumerate(self.factors):
            info[f.name] = torch.mean(rewards_factors[i]).cpu().numpy()

        reward = torch.sum(rewards_factors, dim=0)

        return reward, info

    def reset(self):
        """
        """

        for f in self.factors:
            f.reset()
