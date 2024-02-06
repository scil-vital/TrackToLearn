import os
import nibabel as nib
import numpy as np

from os.path import join as pjoin
from typing import Tuple

from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram
from dipy.io.streamline import save_tractogram

from nibabel.streamlines import Tractogram

from TrackToLearn.environments.backward_tracking_env import \
    BackwardTrackingEnvironment
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.environments.interface_tracking_env import (
    InterfaceNoisyTrackingEnvironment,
    InterfaceTrackingEnvironment)
from TrackToLearn.environments.noisy_tracker import (
    BackwardNoisyTrackingEnvironment,
    NoisyRetrackingEnvironment,
    NoisyTrackingEnvironment)
from TrackToLearn.environments.retracking_env import RetrackingEnvironment
from TrackToLearn.environments.stopping_criteria import (
    is_flag_set, StoppingFlags)
from TrackToLearn.environments.tracking_env import TrackingEnvironment
from TrackToLearn.experiment.experiment import Experiment
from TrackToLearn.utils.utils import LossHistory
from TrackToLearn.utils.comet_monitor import CometMonitor


class TrackToLearnExperiment(Experiment):
    """ Base class for TrackToLearn experiments, even if they're not actually
    RL (such as supervised learning). This "abstract" class provides helper
    methods for loading data, displaying stats and everything that is common
    to all TrackToLearn experiments.
    """

    def run(self):
        """ Abstract version of the main method where data is loaded, classes
        are instanciated, everything is set up.
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
            prefix, self.render)
        print(self.hyperparameters)
        self.comet_monitor.log_parameters(self.hyperparameters)

    def _get_env_dict_and_dto(
        self, interface_tracking_env, no_retrack, noisy
    ) -> Tuple[dict, dict]:
        """ Get the environment class and the environment DTO.

        Parameters
        ----------
        interface_tracking_env: bool
            Whether tracking is done on the interface or not.
        no_retrack: bool
            Whether to retrack or not.
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
            'interface_seeding': self.interface_seeding,
            'fa_map': self.fa_map,
            'n_signal': self.n_signal,
            'n_dirs': self.n_dirs,
            'step_size': self.step_size,
            'theta': self.theta,
            'epsilon': self.epsilon,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'cmc': self.cmc,
            'asymmetric': self.asymmetric,
            'sphere': self.sphere
            if hasattr(self, 'sphere') else None,
            'action_type': self.action_type,
            'noise': self.noise,
            'npv': self.npv,
            'rng': self.rng,
            'alignment_weighting': self.alignment_weighting,
            'straightness_weighting': self.straightness_weighting,
            'length_weighting': self.length_weighting,
            'target_bonus_factor': self.target_bonus_factor,
            'exclude_penalty_factor': self.exclude_penalty_factor,
            'angle_penalty_factor': self.angle_penalty_factor,
            'coverage_weighting': self.coverage_weighting,
            'dense_oracle_weighting': self.dense_oracle_weighting,
            'sparse_oracle_weighting': self.sparse_oracle_weighting,
            'oracle_validator': self.oracle_validator,
            'oracle_stopping_criterion': self.oracle_stopping_criterion,
            'oracle_checkpoint': self.oracle_checkpoint,
            'oracle_filter': self.oracle_filter,
            'scoring_data': self.scoring_data,
            'tractometer_validator': self.tractometer_validator,
            'tractometer_weighting': self.tractometer_weighting,
            'binary_stopping_threshold': self.binary_stopping_threshold,
            'add_neighborhood': self.add_neighborhood,
            'compute_reward': self.compute_reward,
            'device': self.device
        }

        if noisy:
            class_dict = {
                'tracker': NoisyTrackingEnvironment,
                'back_tracker': BackwardNoisyTrackingEnvironment,
                'retracker': NoisyRetrackingEnvironment,
                'interface_tracking_env': InterfaceNoisyTrackingEnvironment
            }
        else:
            class_dict = {
                'tracker': TrackingEnvironment,
                'back_tracker': BackwardTrackingEnvironment,
                'retracker': RetrackingEnvironment,
                'interface_tracking_env': InterfaceTrackingEnvironment
            }
        return class_dict, env_dto

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

        class_dict, env_dto = self._get_env_dict_and_dto(
            self.interface_seeding, self.no_retrack, False)

        # Someone with better knowledge of design patterns could probably
        # clean this
        if self.interface_seeding:
            env = class_dict['interface_tracking_env'].from_dataset(
                env_dto, 'training')
            back_env = None
        else:
            if self.no_retrack:
                env = class_dict['tracker'].from_dataset(env_dto, 'training')
                back_env = class_dict['back_tracker'].from_env(
                    env_dto, env)
            else:
                env = class_dict['tracker'].from_dataset(env_dto, 'training')
                back_env = class_dict['retracker'].from_env(
                    env_dto, env)

        return back_env, env

    def get_valid_envs(self) -> Tuple[BaseEnv, BaseEnv]:
        """ Build environments

        Returns:
        --------
        back_env: BaseEnv
            Backward environment that will be pre-initialized
            with half-streamlines
        env: BaseEnv
            "Forward" environment only initialized with seeds
        """

        class_dict, env_dto = self._get_env_dict_and_dto(
            self.interface_seeding, self.no_retrack, True)

        # Someone with better knowledge of design patterns could probably
        # clean this
        if self.interface_seeding:
            env = class_dict['interface_tracking_env'].from_dataset(
                env_dto, 'training')
            back_env = None
        else:
            if self.no_retrack:
                env = class_dict['tracker'].from_dataset(env_dto, 'training')
                back_env = class_dict['back_tracker'].from_env(
                    env_dto, env)
            else:
                env = class_dict['tracker'].from_dataset(env_dto, 'training')
                back_env = class_dict['retracker'].from_env(
                    env_dto, env)

        return back_env, env

    def get_tracking_envs(self):
        """ Generate environments according to tracking parameters.

        Returns:
        --------
        back_env: BaseEnv
            Backward environment that will be pre-initialized
            with half-streamlines
        env: BaseEnv
            "Forward" environment only initialized with seeds
        """

        class_dict, env_dto = self._get_env_dict_and_dto(
            self.interface_seeding, self.no_retrack, True)

        # Update DTO to include indiv. files instead of hdf5
        env_dto.update({
            'in_odf': self.in_odf,
            'wm_file': self.wm_file,
            'in_seed': self.in_seed,
            'in_mask': self.in_mask,
            'sh_basis': self.sh_basis,
            'input_wm': self.input_wm,
            'reference': self.in_odf,  # reference is inferred from the fODF
            # file instead of being passed directly.
        })

        # Someone with better knowledge of design patterns could probably
        # clean this
        if self.interface_seeding:
            env = class_dict['interface_tracking_env'].from_files(env_dto)
            back_env = None
        else:
            if self.no_retrack:
                env = class_dict['tracker'].from_files(env_dto)
                back_env = class_dict['back_tracker'].from_env(env_dto, env)
            else:
                env = class_dict['tracker'].from_files(env_dto)
                back_env = class_dict['retracker'].from_env(env_dto, env)

        return back_env, env

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

    def score_tractogram(self, filename, affine):
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
            scores = scorer(filename, affine)
            all_scores.update(scores)

        return all_scores

    def save_rasmm_tractogram(
        self,
        tractogram,
        affine: np.ndarray,
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
            self.experiment_path, "tractogram_{}_{}.trk".format(
                self.experiment, self.name))

        # Prune empty streamlines, keep only streamlines that have more
        # than the seed.
        indices = [i for (i, s) in enumerate(tractogram.streamlines)
                   if len(s) > 1]

        tractogram.apply_affine(affine)

        streamlines = tractogram.streamlines[indices]
        data_per_streamline = tractogram.data_per_streamline[indices]
        data_per_point = tractogram.data_per_point[indices]

        sft = StatefulTractogram(
            streamlines,
            reference,
            Space.RASMM,
            origin=Origin.TRACKVIS,
            data_per_streamline=data_per_streamline,
            data_per_point=data_per_point)

        sft.to_rasmm()

        save_tractogram(sft, filename, bbox_valid_check=False)

        return filename

    def log(
        self,
        valid_tractogram: Tractogram,
        env: BaseEnv,
        valid_reward: float = 0,
        i_episode: int = 0,
        scores: dict = None,
        filename: str = None
    ):
        """ Print training infos and log metrics to Comet, if
        activated.

        Parameters
        ----------
        valid_tractogram: Tractogram
            Tractogram generated at validation time.
        env: BaseEnv
            Environment to render the streamlines
        valid_reward: float
            Sum of rewards obtained during validation.
        i_episode: int
            ith training episode.
        scores: dict
            Scores as computed by the tractometer.
        filename:
            Filename to save a screenshot of the rendered environment.
        """
        if valid_tractogram:
            lens = [len(s) for s in valid_tractogram.streamlines]
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

        if self.render:
            # Save image of tractogram to be displayed in comet
            directory = pjoin(self.experiment_path, 'render')
            if not os.path.exists(directory):
                os.makedirs(directory)

            filename = pjoin(
                directory, '{}.png'.format(i_episode))
            env.render(
                valid_tractogram,
                filename)

        if scores is not None:
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
