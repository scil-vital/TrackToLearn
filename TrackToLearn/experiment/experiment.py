import nibabel as nib
import numpy as np

from argparse import ArgumentParser
from os.path import join as pjoin
from typing import Tuple

from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking.streamlinespeed import length

from nibabel.streamlines import Tractogram

from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.environments.tracking_env import (
    TrackingEnvironment)
from TrackToLearn.environments.stopping_criteria import (
    is_flag_set, StoppingFlags)
from TrackToLearn.utils.utils import LossHistory
from TrackToLearn.utils.comet_monitor import CometMonitor


class Experiment(object):
    """ Base class for experiments
    """

    def run(self):
        """ Main method where data is loaded, classes are instanciated,
        everything is set up.
        """
        pass

    def setup_monitors(self):
        #  RL monitors
        self.train_reward_monitor = LossHistory(
            "Train Reward", "train_reward", self.experiment_path)
        self.train_length_monitor = LossHistory(
            "Train Length", "length_reward", self.experiment_path)
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
        if self.tractometer_validator:
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
            prefix)
        print(self.hyperparameters)
        self.comet_monitor.log_parameters(self.hyperparameters)

    def _get_env_dict_and_dto(
        self, noisy
    ) -> Tuple[dict, dict]:
        """ Get the environment class and the environment DTO.

        Parameters
        ----------
        noisy: bool
            Whether to use the noisy environment or not.

        Returns
        -------
        class_dict: dict
            Dictionary of environment classes.
        env_dto: dict
            Dictionary of environment parameters.
        """

        env_dto = {
            'dataset_file': self.dataset_file,
            'fa_map': self.fa_map,
            'n_dirs': self.n_dirs,
            'step_size': self.step_size,
            'theta': self.theta,
            'epsilon': self.epsilon,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'npv': self.npv,
            'rng': self.rng,
            'alignment_weighting': self.alignment_weighting,
            'oracle_bonus': self.oracle_bonus,
            'oracle_validator': self.oracle_validator,
            'oracle_stopping_criterion': self.oracle_stopping_criterion,
            'oracle_checkpoint': self.oracle_checkpoint,
            'scoring_data': self.scoring_data,
            'tractometer_validator': self.tractometer_validator,
            'binary_stopping_threshold': self.binary_stopping_threshold,
            'compute_reward': self.compute_reward,
            'device': self.device,
            'target_sh_order': self.target_sh_order
            if hasattr(self, 'target_sh_order') else None,
        }
        class_dict = {
            'tracking_env': TrackingEnvironment
        }
        return class_dict, env_dto

    def get_env(self) -> Tuple[BaseEnv, BaseEnv]:
        """ Build environments

        Returns:
        --------
        env: BaseEnv
            "Forward" environment only initialized with seeds
        """

        class_dict, env_dto = self._get_env_dict_and_dto(False)

        # Someone with better knowledge of design patterns could probably
        # clean this
        env = class_dict['tracking_env'].from_dataset(
            env_dto, 'training')

        return env

    def get_valid_env(self) -> Tuple[BaseEnv, BaseEnv]:
        """ Build environments

        Returns:
        --------
        env: BaseEnv
            "Forward" environment only initialized with seeds
        """

        class_dict, env_dto = self._get_env_dict_and_dto(True)

        # Someone with better knowledge of design patterns could probably
        # clean this
        env = class_dict['tracking_env'].from_dataset(
            env_dto, 'training')

        return env

    def get_tracking_env(self):
        """ Generate environments according to tracking parameters.

        Returns:
        --------
        env: BaseEnv
            "Forward" environment only initialized with seeds
        """

        class_dict, env_dto = self._get_env_dict_and_dto(True)

        # Update DTO to include indiv. files instead of hdf5
        env_dto.update({
            'in_odf': self.in_odf,
            'wm_file': self.wm_file,
            'in_seed': self.in_seed,
            'in_mask': self.in_mask,
            'sh_basis': self.sh_basis,
            'input_wm': self.input_wm,
            'reference': self.reference_file,
            # file instead of being passed directly.
        })

        # Someone with better knowledge of design patterns could probably
        # clean this
        env = class_dict['tracking_env'].from_files(env_dto)

        return env

    def stopping_stats(self, tractogram):
        """ Compute stopping statistics for a tractogram.

        Parameters
        ----------
        tractogram: Tractogram
            Tractogram to compute statistics on.

        Returns
        -------
        stats: dict
            Dictionary of stopping statistics.
        """
        # Compute stopping statistics
        if tractogram is None:
            return {}
        # Stopping statistics are stored in the data_per_streamline
        # dictionary
        flags = tractogram.data_per_streamline['flags']
        stats = {}
        # Compute the percentage of streamlines that have a given flag set
        # for each flag
        for f in StoppingFlags:
            if len(flags) > 0:
                set_pct = np.mean(is_flag_set(flags, f))
            else:
                set_pct = 0
            stats.update({f.name: set_pct})
        return stats

    def score_tractogram(self, filename, env):
        """ Score a tractogram using the tractometer or the oracle.

        Parameters
        ----------
        filename: str
            Filename of the tractogram to score.
        """
        # Dict of scores
        all_scores = {}

        # Compute scores for the tractogram according
        # to each validator.
        for scorer in self.validators:
            scores = scorer(filename, env)
            all_scores.update(scores)

        return all_scores

    def save_rasmm_tractogram(
        self,
        tractogram,
        subject_id: str,
        env: BaseEnv,
        reference: nib.Nifti1Image
    ) -> str:
        """
        Saves a non-stateful tractogram from the training/validation
        trackers.

        Parameters
        ----------
        tractogram: Tractogram
            Tractogram generated at validation time.

        Returns:
        --------
        filename: str
            Filename of the saved tractogram.
        """

        # Save tractogram so it can be looked at, used by the tractometer
        # and more
        filename = pjoin(
            self.experiment_path, "tractogram_{}_{}_{}.trk".format(
                self.experiment, self.name, subject_id))

        # Prune empty streamlines, keep only streamlines that have more
        # than the seed.
        # indices = [i for (i, s) in enumerate(tractogram.streamlines)
        #            if len(s) > env.min_nb_steps]
        indices = [i for (i, s) in enumerate(tractogram.streamlines)
                   if len(s) > 1]

        tractogram.apply_affine(env.affine_vox2rasmm)

        streamlines = tractogram.streamlines[indices]
        data_per_streamline = tractogram.data_per_streamline[indices]
        data_per_point = tractogram.data_per_point[indices]

        sft = StatefulTractogram(
            streamlines,
            reference,
            Space.RASMM,
            origin=Origin.NIFTI,
            data_per_streamline=data_per_streamline,
            data_per_point=data_per_point)

        sft.to_rasmm()

        save_tractogram(sft, filename, bbox_valid_check=False)

        return filename

    def log(
        self,
        valid_tractogram: Tractogram,
        valid_reward: float = 0,
        i_episode: int = 0,
    ):
        """ Print training infos and log metrics to Comet, if
        activated.

        Parameters
        ----------
        valid_tractogram: Tractogram
            Tractogram generated at validation time.
        valid_reward: float
            Sum of rewards obtained during validation.
        i_episode: int
            ith training episode.
        scores: dict
            Scores as computed by the tractometer.
        """
        if valid_tractogram:
            lens = [length(s) for s in valid_tractogram.streamlines]
        else:
            lens = [0]

        avg_valid_reward = valid_reward / len(lens)
        avg_length = np.mean(lens)  # Euclidian length

        print('---------------------------------------------------')
        print(self.experiment_path)
        print('Episode {} \t avg length: {} \t total reward: {}'.format(
            i_episode,
            avg_length,
            avg_valid_reward))
        print('---------------------------------------------------')

        # Update monitors
        self.len_monitor.update(avg_length)
        self.len_monitor.end_epoch(i_episode)

        self.reward_monitor.update(avg_valid_reward)
        self.reward_monitor.end_epoch(i_episode)

        if self.use_comet and self.comet_experiment is not None:
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


def add_experiment_args(parser: ArgumentParser):
    parser.add_argument('path', type=str,
                        help='Path to experiment')
    parser.add_argument('experiment',
                        help='Name of experiment.')
    parser.add_argument('id', type=str,
                        help='ID of experiment.')
    parser.add_argument('--workspace', type=str, default='BundleTrack',
                        help='Comet.ml workspace')
    parser.add_argument('--rng_seed', default=1337, type=int,
                        help='Seed to fix general randomness')
    parser.add_argument('--use_comet', action='store_true',
                        help='Use comet to display training or not')
    parser.add_argument('--comet_offline_dir', type=str, help='Comet offline '
                        'directory. If enabled, logs will be saved to this '
                        'directory and the experiment will be ran offline.')


def add_data_args(parser: ArgumentParser):
    parser.add_argument('dataset_file',
                        help='Path to preprocessed dataset file (.hdf5)')


def add_environment_args(parser: ArgumentParser):
    parser.add_argument('--n_dirs', default=100, type=int,
                        help='Last n steps taken')
    parser.add_argument(
        '--binary_stopping_threshold',
        type=float, default=0.1,
        help='Lower limit for interpolation of tracking mask value.\n'
        'Tracking will stop below this threshold.')


def add_reward_args(parser: ArgumentParser):
    parser.add_argument('--alignment_weighting', default=1, type=float,
                        help='Alignment weighting for reward')


def add_model_args(parser: ArgumentParser):
    parser.add_argument('--n_actor', default=4096, type=int,
                        help='Number of learners')
    parser.add_argument('--hidden_dims', default='1024-1024-1024', type=str,
                        help='Hidden layers of the model')


def add_tracking_args(parser: ArgumentParser):
    parser.add_argument('--npv', default=2, type=int,
                        help='Number of random seeds per seeding mask voxel.')
    parser.add_argument('--theta', default=30, type=int,
                        help='Max angle between segments for tracking.')
    parser.add_argument('--epsilon', default=90, type=int,
                        help='Max angle between tracking step and fodf peaks.')
    parser.add_argument('--min_length', type=float, default=20.,
                        metavar='m',
                        help='Minimum length of a streamline in mm. '
                        '[%(default)s]')
    parser.add_argument('--max_length', type=float, default=200.,
                        metavar='M',
                        help='Maximum length of a streamline in mm. '
                        '[%(default)s]')
    parser.add_argument('--step_size', default=0.5, type=float,
                        help='Step size for tracking')
    parser.add_argument('--noise', default=0.0, type=float, metavar='sigma',
                        help='Add noise ~ N (0, `noise`) to the agent\'s\n'
                        'output to make tracking more probabilistic.\n'
                        'Should be between 0.0 and 0.1.'
                        '[%(default)s]')


def add_tractometer_args(parser: ArgumentParser):
    tractom = parser.add_argument_group('Tractometer')
    tractom.add_argument('--scoring_data', type=str, default=None,
                         help='Location of the tractometer scoring data.')
    tractom.add_argument('--tractometer_reference', type=str, default=None,
                         help='Reference anatomy for the Tractometer.')
    tractom.add_argument('--tractometer_validator', action='store_true',
                         help='Run tractometer during validation to monitor' +
                         ' how the training is doing w.r.t. ground truth.')
    tractom.add_argument('--tractometer_dilate', default=1, type=int,
                         help='Dilation factor for the ROIs of the '
                         'Tractometer.')


def add_oracle_args(parser: ArgumentParser):
    oracle = parser.add_argument_group('Oracle')
    oracle.add_argument('--oracle_checkpoint', type=str,
                        default='models/tractoracle.ckpt',
                        help='Checkpoint file (.ckpt) of the Oracle')
    oracle.add_argument('--oracle_validator', action='store_true',
                        help='Run a TractOracle model during validation to '
                        'monitor how the training is doing.')
    oracle.add_argument('--oracle_stopping_criterion', action='store_true',
                        help='Stop streamlines according to the Oracle.')
    oracle.add_argument('--oracle_bonus', default=0, type=float,
                        help='Sparse oracle weighting for reward.')
