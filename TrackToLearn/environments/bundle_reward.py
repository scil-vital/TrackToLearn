import numpy as np

from dipy.tracking.stopping_criterion import \
    BinaryStoppingCriterion as DipyStoppingCriterion, \
    StreamlineStatus

from TrackToLearn.environments.reward import Reward


class BundleReward(Reward):

    """ Reward class to compute the bundle reward. Agents will be rewarded
    for reaching bundle endpoints. """

    def __init__(
        self, bundle_endpoint, bundle_mask, min_nb_steps, threshold=0.5
    ):
        """
        Parameters
        ----------
        bundle_endpoint: `numpy.ndarray`
            Bundle endpoints.
        bundle_mask: `numpy.ndarray`
            Binary mask of the bundle.
        min_nb_steps: int
            Minimum number of steps for a streamline to be considered.
        threshold: float
            Threshold to consider a point as part of the bundle.
        """

        self.name = 'bundle_reward'

        self.N = bundle_endpoint.shape[-1]
        self.min_nb_steps = min_nb_steps

        self.endpoint_criterion = [
            DipyStoppingCriterion(bundle_endpoint[..., i])
            for i in range(self.N)]

        self.threshold = threshold

    def __call__(self, streamlines, bundle_idx, dones):
        """ Compute the reward for each streamline.

        Parameters
        ----------
        streamlines: `numpy.ndarray`
            Streamlines to compute the reward for.

        Returns
        -------
        rewards: `numpy.ndarray`
            Rewards for each streamline.
        """
        reward = np.zeros(len(streamlines), dtype=bool)
        head_tail = np.zeros(len(streamlines), dtype=bool)

        L = streamlines.shape[1]
        if L >= self.min_nb_steps:
            for i in range(self.N):
                b_i = np.arange(len(streamlines))[bundle_idx == i]
                for j, s in enumerate(streamlines[b_i]):
                    point = s[-1].astype(np.double)
                    status = self.endpoint_criterion[i].check_point(point)
                    if status == StreamlineStatus.TRACKPOINT:
                        head_tail[b_i[j]] = True

        reward = head_tail

        return reward * dones
