import numpy as np

from collections import defaultdict
from tqdm import tqdm
from typing import Tuple

from dipy.tracking.streamlinespeed import compress_streamlines
from nibabel.streamlines import Tractogram
from nibabel.streamlines.tractogram import LazyTractogram
from nibabel.streamlines.tractogram import TractogramItem

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.algorithms.shared.utils import add_to_means
from TrackToLearn.environments.env import BaseEnv


class Tracker(object):
    """ Tracking class similar to scilpy's or dwi_ml's. This class is
    responsible for generating streamlines, as well as giving back training
    or RL-associated metrics if applicable.
    """

    def __init__(
        self,
        alg: RLAlgorithm,
        env: BaseEnv,
        back_env: BaseEnv,
        n_actor: int,
        interface_seeding: bool,
        no_retrack: bool,
        compress: float = 0.0,
        save_seeds: bool = False
    ):
        """

        Parameters
        ----------
        alg: RLAlgorithm
            Tracking agent.
        env: BaseEnv
            Forward environment to track.
        back_env: BaseEnv
            Backward environment to track.
        compress: float
            Compression factor when saving streamlines.

        """

        self.alg = alg
        self.env = env
        self.back_env = back_env
        self.n_actor = n_actor
        self.interface_seeding = interface_seeding
        self.no_retrack = no_retrack
        self.compress = compress
        self.save_seeds = save_seeds

    def track(
        self,
    ):
        """ Actual tracking function. Use this if you just want streamlines.

        Track with a generator to save streamlines to file
        as they are tracked. Used at tracking (test) time. No
        reward should be computed.

        Returns:
        --------
        tractogram: Tractogram
            Tractogram in a generator format.

        """

        # Presume iso vox
        vox_size = abs(self.env.affine_vox2rasmm[0][0])

        compress_th_vox = self.compress / vox_size

        batch_size = self.n_actor

        # Shuffle seeds so that massive tractograms wont load "sequentially"
        # when partially displayed
        np.random.shuffle(self.env.seeds)

        def tracking_generator():
            # Switch policy to eval mode so no gradients are computed
            self.alg.policy.eval()
            # Track for every seed in the environment
            for i, start in enumerate(
                tqdm(range(0, len(self.env.seeds), batch_size))
            ):

                # Last batch might not be "full"
                end = min(start + batch_size, len(self.env.seeds))

                state = self.env.reset(start, end)

                # Track forward
                self.alg.validation_episode(
                    state, self.env)
                batch_tractogram = self.env.get_streamlines()

                if not self.interface_seeding:
                    state = self.back_env.reset(batch_tractogram.streamlines)

                    # Track backwards
                    self.alg.validation_episode(
                        state, self.back_env)
                    batch_tractogram = self.back_env.get_streamlines()

                for item in batch_tractogram:

                    streamline_length = len(item)

                    streamline = item.streamline
                    streamline += 0.5
                    streamline *= vox_size

                    seed_dict = {}
                    if self.save_seeds:
                        seed = item.data_for_streamline['seeds']
                        seed_dict = {'seeds': seed-0.5}

                    if self.compress:
                        streamline = compress_streamlines(
                            streamline, compress_th_vox)

                    if (self.env.min_nb_steps < streamline_length <
                            self.env.max_nb_steps):
                        yield TractogramItem(
                            streamline, seed_dict, {})

        tractogram = LazyTractogram.from_data_func(tracking_generator)

        return tractogram

    def track_and_train(
        self,
    ) -> Tuple[Tractogram, float, float, float]:
        """
        Call the main training loop forward then backward.
        This can be considered an "epoch". Note that N=self.n_actor
        streamlines will be tracked instead of one streamline per seed.

        Returns
        -------
        streamlines: Tractogram
            Tractogram containing the tracked streamline
        losses: dict
            Dictionary containing various losses and metrics
            w.r.t the agent's training.
        running_reward: float
            Cummulative training steps reward
        """

        self.alg.policy.train()

        mean_losses = defaultdict(list)

        # Fetch n=n_actor seeds
        state = self.env.nreset(self.n_actor)

        # Track and train forward
        reward, losses, length = \
            self.alg._episode(state, self.env)
        # Get the streamlines generated from forward training
        train_tractogram = self.env.get_streamlines()

        mean_losses = add_to_means(mean_losses, losses)

        if not self.interface_seeding:
            # Flip streamlines to initialize backwards tracking
            state = self.back_env.reset(train_tractogram.streamlines)

            # Track and train backwards
            back_reward, losses, length = \
                self.alg._episode(state, self.back_env)
            # Get the streamlines generated from backward training
            train_tractogram = self.back_env.get_streamlines()

            mean_losses = add_to_means(mean_losses, losses)

            # Retracking also rewards the agents
            if self.no_retrack:
                reward += back_reward
            else:
                reward = back_reward

        return (
            train_tractogram,
            mean_losses,
            reward)

    def track_and_validate(
        self,
    ) -> Tuple[Tractogram, float, dict]:
        """
        Run the tracking algorithm without training to see how it performs, but
        still compute the reward.

        Returns:
        --------
        tractogram: Tractogram
            Validation tractogram.
        reward: float
            Reward obtained during validation.
        """
        # Switch policy to eval mode so no gradients are computed
        self.alg.policy.eval()

        # Initialize tractogram
        tractogram = None

        # Reward gotten during validation
        cummulative_reward = 0

        def _generate_streamlines_and_rewards():

            # Track for every seed in the environment
            for i, start in enumerate(
                    tqdm(range(0, len(self.env.seeds), self.n_actor))):

                # Last batch might not be "full"
                end = min(start + self.n_actor, len(self.env.seeds))

                state = self.env.reset(start, end)

                # Track forward
                reward = self.alg.validation_episode(state, self.env)
                batch_tractogram = self.env.get_streamlines()

                if not self.interface_seeding:
                    # Initialize backwards tracking
                    state = self.back_env.reset(batch_tractogram.streamlines)

                    # Track backwards
                    reward = self.alg.validation_episode(
                        state, self.back_env)
                    batch_tractogram = self.back_env.get_streamlines()

                yield batch_tractogram, reward

        for t, r in _generate_streamlines_and_rewards():
            if tractogram is None:
                tractogram = t
            else:
                tractogram += t
            cummulative_reward += r

        return tractogram,  cummulative_reward
