import os
import numpy as np

from os.path import join as pjoin
from typing import Tuple

from challenge_scoring.metrics.scoring import score_submission
from challenge_scoring.utils.attributes import load_attribs
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking.metrics import length as slength
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
from TrackToLearn.environments.tracking_env import TrackingEnvironment
from TrackToLearn.experiment.experiment import Experiment
from TrackToLearn.utils.utils import LossHistory
from TrackToLearn.utils.comet_monitor import CometMonitor


class TrackToLearnExperiment(Experiment):
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
            self.comet_experiment, self.id, self.experiment_path,
            prefix, self.render)

        self.comet_monitor.log_parameters(self.hyperparameters)

    def _get_env_dict_and_dto(
        self, interface_tracking_env, no_retrack, noisy
    ) -> Tuple[dict, dict]:

        env_dto = {
            'dataset_file': self.dataset_file,
            'subject_id': self.subject_id,
            'interface_seeding': self.interface_seeding,
            'fa_map': self.fa_map,
            'n_signal': self.n_signal,
            'n_dirs': self.n_dirs,
            'step_size': self.step_size,
            'theta': self.theta,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'cmc': self.cmc,
            'asymmetric': self.asymmetric,
            'prob': self.prob,
            'npv': self.npv,
            'rng': self.rng,
            'scoring_data': self.scoring_data,
            'reference': self.reference_file,
            'alignment_weighting': self.alignment_weighting,
            'straightness_weighting': self.straightness_weighting,
            'length_weighting': self.length_weighting,
            'target_bonus_factor': self.target_bonus_factor,
            'exclude_penalty_factor': self.exclude_penalty_factor,
            'angle_penalty_factor': self.angle_penalty_factor,
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
                env_dto, 'validation')
            back_env = None
        else:
            if self.no_retrack:
                env = class_dict['tracker'].from_dataset(env_dto, 'validation')
                back_env = class_dict['back_tracker'].from_env(
                    env_dto, env)
            else:
                env = class_dict['tracker'].from_dataset(env_dto, 'validation')
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

    def score_tractogram(self, tractogram):

        #  Load bundle attributes for tractometer
        # TODO: No need to load this every time, should only be loaded
        # once
        gt_bundles_attribs_path = pjoin(
            self.scoring_data, 'gt_bundles_attributes.json')
        basic_bundles_attribs = load_attribs(gt_bundles_attribs_path)

        filename = self.save_vox_tractogram(tractogram)

        # Score tractogram
        scores = score_submission(
            filename,
            self.scoring_data,
            basic_bundles_attribs,
            compute_ic_ib=True)

        return scores

    def save_vox_tractogram(
        self,
        tractogram,
    ) -> str:
        """
        Saves a tractogram into vox space as generated by Track-to-Learn.

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
                self.experiment, self.id, self.valid_subject_id))

        # Prune empty streamlines, keep only streamlines that have more
        # than the seed.
        indices = [i for (i, s) in enumerate(tractogram.streamlines)
                   if len(s) > 1]
        streamlines = tractogram.streamlines[indices]
        data_per_streamline = tractogram.data_per_streamline[indices]
        data_per_point = tractogram.data_per_point[indices]

        sft = StatefulTractogram(
            streamlines,
            self.reference_file,
            Space.VOX,
            data_per_streamline=data_per_streamline,
            data_per_point=data_per_point)

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

        lens = [slength(s) for s in valid_tractogram.streamlines]
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
