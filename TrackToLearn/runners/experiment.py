#!/usr/bin/env python
import os
import numpy as np
import torch

from os.path import join as pjoin
from typing import Tuple

from challenge_scoring.metrics.scoring import score_submission
from challenge_scoring.utils.attributes import load_attribs
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking.metrics import length as slength
from nibabel.streamlines import Tractogram

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.environments.noisy_tracker import (
    BackwardNoisyTracker,
    NoisyTracker)
from TrackToLearn.environments.tracker import (
    BackwardTracker,
    Tracker)
from TrackToLearn.utils.utils import LossHistory
from TrackToLearn.utils.comet_monitor import CometMonitor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())


class TrackToLearnExperiment(object):
    """ Base class for TrackToLearn experiments, even if they're not actually
    RL (such as supervised learning). This "abstract" class provides helper
    methods for loading data, displaying stats and everything that is common
    to all TrackToLearn experiments
    """

    def run(self):
        """ Main method where data is loaded, classes are instanciated,
        everything is set up.
        """
        pass

    def setup_monitors(self):
        #  RL monitors
        self.train_reward_monitor = LossHistory(
            "Train Reward - Alignment", "train_reward", self.experiment_path)
        self.reward_monitor = LossHistory(
            "Reward - Alignment", "reward", self.experiment_path)
        self.actor_loss_monitor = LossHistory(
            "Loss - Actor Policy Loss", "actor_loss", self.experiment_path)
        self.critic_loss_monitor = LossHistory(
            "Loss - Critic MSE Loss", "critic_loss", self.experiment_path)
        self.len_monitor = LossHistory(
            "Length", "length", self.experiment_path)

        # Tractometer monitors
        # TODO: Infer the number of bundles from the GT
        if self.run_tractometer:
            self.vc_monitor = LossHistory(
                "Valid Connections", "vc", self.experiment_path)
            self.ic_monitor = LossHistory(
                "Invalid Connections", "ic", self.experiment_path)
            self.nc_monitor = LossHistory(
                "Non-Connections", "nc", self.experiment_path)
            self.vb_monitor = LossHistory(
                "Valid Bundles", "VB", self.experiment_path)
            self.ib_monitor = LossHistory(
                "Invalid Bundles", "IB", self.experiment_path)
            self.ol_monitor = LossHistory(
                "Overlap monitor", "ol", self.experiment_path)

        else:
            self.vc_monitor = None
            self.ic_monitor = None
            self.nc_monitor = None
            self.vb_monitor = None
            self.ib_monitor = None
            self.ol_monitor = None

        # SL monitors
        self.pretrain_actor_monitor = LossHistory(
            'Pretraining Actor Loss', 'pretrain_actor_loss',
            self.experiment_path)
        self.pretrain_critic_monitor = LossHistory(
            'Pretraining Critic Loss', 'pretrain_critic_loss',
            self.experiment_path)

        # Initialize monitors here as the first pass won't include losses
        self.actor_loss_monitor.update(0)
        self.actor_loss_monitor.end_epoch(0)
        self.critic_loss_monitor.update(0)
        self.critic_loss_monitor.end_epoch(0)

    def setup_comet(self, prefix=''):
        """ Setup comet environment
        """
        # The comet object that will handle monitors
        self.comet_monitor = CometMonitor(
            self.comet_experiment, self.name, self.experiment_path,
            prefix, self.render)

        self.comet_monitor.log_parameters(self.hyperparameters)

    def get_envs(self) -> Tuple[BaseEnv, BaseEnv]:
        """ Build environments

        Returns:
        --------
        back_env: BaseEnv
            Backward environment that will be pre-initialized
            with half-streamlines
        env: BaseEnv
            "Forward" environment only initialized with seeds
        """

        # Not sure if parameters should come from `self` of actual
        # function parameters. It feels a bit dirty to have everything
        # in `self`, but then there's a crapload of parameters
        # Backward environment
        back_env = BackwardTracker(
            self.dataset_file,
            self.subject_id,
            self.n_signal,
            self.n_dirs,
            self.step_size,
            self.max_angle,
            self.min_length,
            self.max_length,
            self.n_seeds_per_voxel,
            self.rng,
            self.add_neighborhood,
            True,
            device)

        # Forward environment
        env = Tracker(
            self.dataset_file,
            self.subject_id,
            self.n_signal,
            self.n_dirs,
            self.step_size,
            self.max_angle,
            self.min_length,
            self.max_length,
            self.n_seeds_per_voxel,
            self.rng,
            self.add_neighborhood,
            True,
            device)

        return back_env, env

    def get_test_envs(self) -> Tuple[BaseEnv, BaseEnv]:
        """ Build environments

        Returns:
        --------
        back_env: BaseEnv
            Backward environment that will be pre-initialized
            with half-streamlines
        env: BaseEnv
            "Forward" environment only initialized with seeds
        """

        # Not sure if parameters should come from `self` of actual
        # function parameters. It feels a bit dirty to have everything
        # in `self`, but then there's a crapload of parameters
        # Backward environment
        back_env = BackwardNoisyTracker(
            self.test_dataset_file,
            self.test_subject_id,
            self.fa_map,
            self.n_signal,
            self.n_dirs,
            self.step_size,
            self.max_angle,
            self.min_length,
            self.max_length,
            self.n_seeds_per_voxel,
            self.rng,
            self.add_neighborhood,
            self.valid_noise,
            self.compute_reward,
            device)

        # Forward environment
        env = NoisyTracker(
            self.test_dataset_file,
            self.test_subject_id,
            self.fa_map,
            self.n_signal,
            self.n_dirs,
            self.step_size,
            self.max_angle,
            self.min_length,
            self.max_length,
            self.n_seeds_per_voxel,
            self.rng,
            self.add_neighborhood,
            self.valid_noise,
            self.compute_reward,
            device)

        return back_env, env

    def test(
        self,
        alg: RLAlgorithm,
        env: BaseEnv,
        back_env: BaseEnv,
        save_model: bool = True,
    ) -> Tuple[Tractogram, float]:
        """
        Run the tracking algorithm without noise to see how it performs

        Parameters
        ----------
        alg: RLAlgorithm
            Tracking algorithm that contains the being-trained policy
        env: BaseEnv
            Forward environment
        back_env: BaseEnv
            Backward environment
        save_model: bool
            Save the model or not

        Returns:
        --------
        tractogram: Tractogram
            validation tractogram
        reward: float
            Reward obtained during validation
        """

        # Save the model so it can be loaded by the tracking
        if save_model:
            directory = os.path.dirname(pjoin(self.experiment_path, "model"))
            if not os.path.exists(directory):
                os.makedirs(directory)
            alg.policy.save(directory, "last_model_state")

        # Launch the tracking
        tractogram, reward = alg.run_validation(
            self.tracking_batch_size,
            env,
            back_env)

        return tractogram, reward

    def _save_tractogram(
        self,
        tractogram,
        reference,
        space,
        path
    ):

        indices = [i for (i, s) in enumerate(tractogram.streamlines)
                   if len(s) > 1]
        streamlines = tractogram.streamlines[indices]
        data_per_streamline = tractogram.data_per_streamline[indices]
        data_per_point = tractogram.data_per_point[indices]

        sft = StatefulTractogram(
            streamlines,
            reference,
            space,
            data_per_streamline=data_per_streamline,
            data_per_point=data_per_point)

        save_tractogram(sft, path, bbox_valid_check=False)

    def display(
        self,
        valid_tractogram: Tractogram,
        env: BaseEnv,
        valid_reward: float = 0,
        i_episode: int = 0,
        run_tractometer: bool = False,
    ):
        """
        Stats stuff

        There's so much going on in this function, it should be split or
        something

        Parameters
        ----------
        valid_tractogram: Tractogram
            Tractogram containing all the streamlines tracked during the last
            validation run
        env: BaseEnv
            Environment used to render streamlines
        valid_reward: np.ndarray of float of size
            Reward of the last validation run
        i_episode: int
            Current episode
        """

        lens = [slength(s) for s in valid_tractogram.streamlines]

        avg_length = np.mean(lens)  # Euclidian length

        print('---------------------------------------------------')
        print(self.experiment_path)
        print('Episode {} \t avg length: {} \t total reward: {}'.format(
            i_episode,
            avg_length,
            valid_reward))
        print('---------------------------------------------------')

        # Save tractogram so it can be looked at, used by the tractometer
        # and more
        filename = pjoin(self.experiment_path,
                         "tractogram_{}_{}_{}.trk".format(
                             self.experiment, self.name, self.test_subject_id))

        self._save_tractogram(
            valid_tractogram,
            self.reference_file,
            Space.VOX,
            filename)

        if self.comet_experiment is not None:
            if self.run_tractometer and run_tractometer:
                #  Load bundle attributes for tractometer
                # TODO: No need to load this every time, should only be loaded
                # once
                gt_bundles_attribs_path = pjoin(
                    self.ground_truth_folder, 'gt_bundles_attributes.json')
                basic_bundles_attribs = load_attribs(gt_bundles_attribs_path)

                # Score tractogram
                scores = score_submission(
                    filename,
                    {'orientation': 'unknown'},
                    self.ground_truth_folder,
                    basic_bundles_attribs)

                self.vc_monitor.update(scores['VC'])
                self.ic_monitor.update(scores['IC'])
                self.nc_monitor.update(scores['NC'])
                self.vb_monitor.update(scores['VB'])
                self.ib_monitor.update(scores['IB'])
                self.ol_monitor.update(scores['mean_OL'])

                self.vc_monitor.end_epoch(i_episode)
                self.ic_monitor.end_epoch(i_episode)
                self.nc_monitor.end_epoch(i_episode)
                self.vb_monitor.end_epoch(i_episode)
                self.ib_monitor.end_epoch(i_episode)
                self.ol_monitor.end_epoch(i_episode)

            if self.render:
                # Save image of tractogram to be displayed in comet
                env.render(
                    valid_tractogram,
                    '{}.png'.format(i_episode))

            # Update monitors
            self.len_monitor.update(avg_length)
            self.len_monitor.end_epoch(i_episode)

            self.reward_monitor.update(valid_reward)
            self.reward_monitor.end_epoch(i_episode)

            # Update comet
            self.comet_monitor.update(
                self.reward_monitor,
                self.len_monitor,
                self.vc_monitor,
                self.ic_monitor,
                self.nc_monitor,
                self.vb_monitor,
                self.ib_monitor,
                self.ol_monitor,
                i_episode=i_episode)


def add_experiment_args(parser):
    parser.add_argument('path', type=str,
                        help='Path to experiment')
    parser.add_argument('experiment',
                        help='Type of experiment (usually alg + dataset). ' +
                        'Will group runs of same experiment but different ' +
                        'parameters together')
    parser.add_argument('name', type=str,
                        help='Name to experiment')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use gpu or not')
    parser.add_argument('--rng_seed', default=1337, type=int,
                        help='Seed to fix general randomness')
    parser.add_argument('--use_comet', action='store_true',
                        help='Use comet to display training or not')
    parser.add_argument('--run_tractometer', action='store_true',
                        help='Run tractometer during validation to monitor' +
                        ' how the training is doing w.r.t. ground truth')
    parser.add_argument('--render', action='store_true',
                        help='Save screenshots of tracking as it goes along.' +
                        'Preferably disabled on non-graphical environments')


def add_data_args(parser):
    parser.add_argument('dataset_file',
                        help='Path to preprocessed dataset file (.hdf5)')
    parser.add_argument('subject_id',
                        help='Subject id to fetch from the dataset file')
    parser.add_argument('test_dataset_file',
                        help='Path to preprocessed dataset file (.hdf5)')
    parser.add_argument('test_subject_id',
                        help='Subject id to fetch from the dataset file')
    parser.add_argument('reference_file',
                        help='Path to binary seeding mask (.nii|.nii.gz)')
    parser.add_argument('ground_truth_folder',
                        help='Path to reference streamlines (.nii|.nii.gz)')


def add_environment_args(parser):
    parser.add_argument('--n_signal', default=1, type=int,
                        help='Signal at the last n positions')
    parser.add_argument('--n_dirs', default=4, type=int,
                        help='Last n steps taken')
    parser.add_argument('--add_neighborhood', default=1, type=float,
                        help='Add neighborhood to model input')
    parser.add_argument('--n_seeds_per_voxel', default=2, type=int,
                        help='Number of random seeds per seeding mask voxel')
    parser.add_argument('--max_angle', default=45, type=int,
                        help='Max angle for tracking')
    parser.add_argument('--min_length', default=20, type=int,
                        help='Minimum length for tracts')
    parser.add_argument('--max_length', default=200, type=int,
                        help='Maximum length for tracts')
    parser.add_argument('--step_size', default=0.75, type=float,
                        help='Step size for tracking')


def add_model_args(parser):
    parser.add_argument('--n_latent_var', default=1024, type=int,
                        help='Width of layers for the model')
    parser.add_argument('--hidden_layers', default=3, type=int,
                        help='Depth of layers for the model')
    parser.add_argument('--load_policy', default=None, type=str,
                        help='Path to pretrained model')


def add_tracking_args(parser):
    parser.add_argument('--tracking_batch_size', default=10000, type=int,
                        help='Number of seeds used per episode')
    parser.add_argument('--valid_noise', default=0.010, type=float,
                        help='Noise to make a probablistic output during '
                        'test/valid')
