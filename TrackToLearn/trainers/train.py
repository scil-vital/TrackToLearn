import json
import os
import random
from os.path import join as pjoin

import numpy as np
import torch

from dipy.tracking.streamlinespeed import length

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.algorithms.shared.utils import mean_losses, mean_rewards
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.experiment.experiment import (add_data_args,
                                                add_environment_args,
                                                add_experiment_args,
                                                add_model_args,
                                                add_oracle_args,
                                                add_reward_args,
                                                add_tracking_args,
                                                add_tractometer_args)
from TrackToLearn.experiment.oracle_validator import OracleValidator
from TrackToLearn.experiment.tractometer_validator import TractometerValidator
from TrackToLearn.experiment.experiment import Experiment
from TrackToLearn.tracking.tracker import Tracker
from TrackToLearn.utils.torch_utils import get_device, assert_accelerator


class TrackToLearnTraining(Experiment):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        train_dto: dict,
        comet_experiment,
    ):
        """
        Parameters
        ----------
        train_dto: dict
            Dictionnary containing the training parameters.
            Put into a dictionnary to prevent parameter errors if modified.
        """
        # TODO: Find a better way to pass parameters around

        # Experiment parameters
        self.experiment_path = train_dto['path']
        self.experiment = train_dto['experiment']
        self.name = train_dto['id']

        # RL parameters
        self.max_ep = train_dto['max_ep']
        self.log_interval = train_dto['log_interval']
        self.noise = train_dto['noise']

        # Training parameters
        self.lr = train_dto['lr']
        self.gamma = train_dto['gamma']

        #  Tracking parameters
        self.step_size = train_dto['step_size']
        self.dataset_file = train_dto['dataset_file']
        self.rng_seed = train_dto['rng_seed']
        self.npv = train_dto['npv']

        # Angular thresholds
        self.theta = train_dto['theta']
        self.epsilon = train_dto['epsilon']

        # More tracking parameters
        self.min_length = train_dto['min_length']
        self.max_length = train_dto['max_length']
        self.binary_stopping_threshold = train_dto['binary_stopping_threshold']

        # Reward parameters
        self.alignment_weighting = train_dto['alignment_weighting']

        # Model parameters
        self.hidden_dims = train_dto['hidden_dims']

        # Environment parameters
        self.n_actor = train_dto['n_actor']
        self.n_dirs = train_dto['n_dirs']

        # Oracle parameters
        self.oracle_checkpoint = train_dto['oracle_checkpoint']
        self.oracle_bonus = train_dto['oracle_bonus']
        self.oracle_validator = train_dto['oracle_validator']
        self.oracle_stopping_criterion = train_dto['oracle_stopping_criterion']

        # Tractometer parameters
        self.tractometer_validator = train_dto['tractometer_validator']
        self.tractometer_dilate = train_dto['tractometer_dilate']
        self.tractometer_reference = train_dto['tractometer_reference']
        self.scoring_data = train_dto['scoring_data']

        self.compute_reward = True  # Always compute reward during training
        self.fa_map = None

        # Various parameters
        self.comet_experiment = comet_experiment
        self.last_episode = 0

        self.device = get_device()

        self.use_comet = train_dto['use_comet']

        # RNG
        torch.manual_seed(self.rng_seed)
        np.random.seed(self.rng_seed)
        self.rng = np.random.RandomState(seed=self.rng_seed)
        random.seed(self.rng_seed)

        directory = pjoin(self.experiment_path, 'model')
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.hyperparameters = {
            # RL parameters
            # TODO: Make sure all parameters are logged
            'name': self.name,
            'experiment': self.experiment,
            'max_ep': self.max_ep,
            'log_interval': self.log_interval,
            'lr': self.lr,
            'gamma': self.gamma,
            # Data parameters
            'step_size': self.step_size,
            'random_seed': self.rng_seed,
            'dataset_file': self.dataset_file,
            'n_seeds_per_voxel': self.npv,
            'max_angle': self.theta,
            'epsilon': self.epsilon,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'binary_stopping_threshold': self.binary_stopping_threshold,
            # Model parameters
            'experiment_path': self.experiment_path,
            'hidden_dims': self.hidden_dims,
            'last_episode': self.last_episode,
            'n_actor': self.n_actor,
            'n_dirs': self.n_dirs,
            'noise': self.noise,
            # Reward parameters
            'alignment_weighting': self.alignment_weighting,
            # Oracle parameters
            'oracle_bonus': self.oracle_bonus,
            'oracle_checkpoint': self.oracle_checkpoint,
            'oracle_stopping_criterion': self.oracle_stopping_criterion,
        }

    def save_hyperparameters(self):
        """ Save hyperparameters to json file
        """
        # Add input and action size to hyperparameters
        # These are added here because they are not known before
        self.hyperparameters.update({'input_size': self.input_size,
                                     'action_size': self.action_size,
                                     'voxel_size': str(self.voxel_size),
                                     'target_sh_order': self.target_sh_order
                                     })

        directory = pjoin(self.experiment_path, "model")
        with open(
            pjoin(directory, "hyperparameters.json"),
            'w'
        ) as json_file:
            json_file.write(
                json.dumps(
                    self.hyperparameters,
                    indent=4,
                    separators=(',', ': ')))

    def save_model(self, alg):
        """ Save the model state to disk
        """

        directory = pjoin(self.experiment_path, "model")
        if not os.path.exists(directory):
            os.makedirs(directory)
        alg.agent.save(directory, "last_model_state")

    def rl_train(
        self,
        alg: RLAlgorithm,
        env: BaseEnv,
        valid_env: BaseEnv,
    ):
        """ Train the RL algorithm for N epochs. An epoch here corresponds to
        running tracking on the training set until all streamlines are done.
        This loop should be algorithm-agnostic. Between epochs, report stats
        so they can be monitored during training

        Parameters:
        -----------
            alg: RLAlgorithm
                The RL algorithm, either TD3, PPO or any others
            env: BaseEnv
                The tracking environment
            valid_env: BaseEnv
                The validation tracking environment (forward).
            """

        # Current epoch
        i_episode = 0
        # Transition counter
        t = 0

        # Initialize Trackers, which will handle streamline generation and
        # trainnig
        train_tracker = Tracker(
            alg, self.n_actor, prob=0.0, compress=0.0)

        valid_tracker = Tracker(
            alg, self.n_actor,
            prob=1.0, compress=0.0)

        # Setup validators, which will handle validation and scoring
        # of the generated streamlines
        self.validators = []
        if self.tractometer_validator:
            self.validators.append(TractometerValidator(
                self.scoring_data, self.tractometer_reference,
                dilate_endpoints=self.tractometer_dilate))
        if self.oracle_validator:
            self.validators.append(OracleValidator(
                self.oracle_checkpoint, self.device))

        # Run tracking before training to see what an untrained network does
        valid_env.load_subject()
        valid_tractogram, valid_reward = valid_tracker.track_and_validate(
            valid_env)
        stopping_stats = self.stopping_stats(valid_tractogram)
        print(stopping_stats)
        if valid_tractogram:
            if self.use_comet:
                self.comet_monitor.log_losses(stopping_stats, i_episode)

            filename = self.save_rasmm_tractogram(valid_tractogram,
                                                  valid_env.subject_id,
                                                  valid_env,
                                                  valid_env.reference)
            scores = self.score_tractogram(filename, valid_env)
            print({k: v for k, v in scores.items() if type(v) is not tuple})

            if self.use_comet:
                self.comet_monitor.log_losses(scores, i_episode)
        self.save_model(alg)

        # Display the results of the untrained network
        self.log(
            valid_tractogram, valid_reward, i_episode)

        # Main training loop
        while i_episode < self.max_ep:

            # Last episode/epoch. Was initially for resuming experiments but
            # since they take so little time I just restart them from scratch
            # Not sure what to do with this
            self.last_episode = i_episode

            # Train for an episode
            env.load_subject()
            tractogram, losses, reward, reward_factors = \
                train_tracker.track_and_train(env)

            # Compute average streamline length
            lengths = length(tractogram.streamlines)
            avg_length = np.mean(lengths)  # Nb. of steps

            # Keep track of how many transitions were gathered
            t += sum(lengths)

            # Compute average reward per streamline
            # Should I use the mean or the sum ?
            avg_reward = reward / self.n_actor

            print(
                f"Episode Num: {i_episode+1} "
                f"Avg len: {avg_length:.3f} Avg. reward: "
                f"{avg_reward:.3f} sub: {env.subject_id}")

            # Update monitors
            self.train_reward_monitor.update(avg_reward)
            self.train_reward_monitor.end_epoch(i_episode)
            self.train_length_monitor.update(avg_length)
            self.train_length_monitor.end_epoch(i_episode)

            i_episode += 1
            # Update comet logs
            if self.use_comet and self.comet_experiment is not None:
                mean_ep_reward_factors = mean_rewards(reward_factors)
                self.comet_monitor.log_losses(
                    mean_ep_reward_factors, i_episode)

                self.comet_monitor.update_train(
                    self.train_reward_monitor, i_episode)
                self.comet_monitor.update_train(
                    self.train_length_monitor, i_episode)
                mean_ep_losses = mean_losses(losses)
                self.comet_monitor.log_losses(mean_ep_losses, i_episode)

            # Time to do a valid run and display stats
            if i_episode % self.log_interval == 0:
                # Validation run
                valid_env.load_subject()
                valid_tractogram, valid_reward = \
                    valid_tracker.track_and_validate(valid_env)
                stopping_stats = self.stopping_stats(valid_tractogram)
                print(stopping_stats)

                if self.use_comet:
                    self.comet_monitor.log_losses(stopping_stats, i_episode)
                filename = self.save_rasmm_tractogram(
                    valid_tractogram, valid_env.subject_id,
                    valid_env, valid_env.reference)
                scores = self.score_tractogram(
                    filename, valid_env)
                print(
                    {k: v for k, v in scores.items() if type(v) is not tuple})

                # Display what the network is capable-of "now"
                self.log(
                    valid_tractogram, valid_reward, i_episode)
                if self.use_comet:
                    self.comet_monitor.log_losses(scores, i_episode)
                self.save_model(alg)

        # End of training, save the model and hyperparameters and track
        valid_env.load_subject()
        valid_tractogram, valid_reward = valid_tracker.track_and_validate(
            valid_env)
        stopping_stats = self.stopping_stats(valid_tractogram)
        print(stopping_stats)

        if self.use_comet:
            self.comet_monitor.log_losses(stopping_stats, i_episode)

        filename = self.save_rasmm_tractogram(valid_tractogram,
                                              valid_env.subject_id,
                                              valid_env,
                                              valid_env.reference)
        scores = self.score_tractogram(filename, valid_env)
        print({k: v for k, v in scores.items() if type(v) is not tuple})

        # Display what the network is capable-of "now"
        self.log(
            valid_tractogram, valid_reward, i_episode)

        if self.use_comet:
            self.comet_monitor.log_losses(scores, i_episode)

        self.save_model(alg)

    def run(self):
        """ Prepare the environment, algorithm and trackers and run the
        training loop
        """

        assert_accelerator(), \
            "Training is only supported with hardware accelerated devices."

        # Instantiate environment. Actions will be fed to it and new
        # states will be returned. The environment updates the streamline
        # internally
        env = self.get_env()
        valid_env = self.get_valid_env()

        # Get example state to define NN input size
        self.input_size = env.get_state_size()
        self.action_size = env.get_action_size()

        # Voxel size
        self.voxel_size = env.get_voxel_size()
        # SH Order (used for tracking afterwards)
        self.target_sh_order = env.target_sh_order

        max_traj_length = env.max_nb_steps

        # The RL training algorithm
        alg = self.get_alg(max_traj_length)

        # Save hyperparameters
        self.save_hyperparameters()

        # Setup monitors to monitor training as it goes along
        self.setup_monitors()

        # Setup comet monitors to monitor experiment as it goes along
        if self.use_comet:
            self.setup_comet()

        # Start training !
        self.rl_train(alg, env, valid_env)


def add_rl_args(parser):
    # Add RL training arguments.
    parser.add_argument('--max_ep', default=1000, type=int,
                        help='Number of episodes to run the training '
                        'algorithm')
    parser.add_argument('--log_interval', default=10, type=int,
                        help='Log statistics, update comet, save the model '
                        'and hyperparameters at n steps')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='Learning rate')
    parser.add_argument('--gamma', default=0.95, type=float,
                        help='Gamma param for reward discounting')

    add_reward_args(parser)


def add_training_args(parser):
    # Add all training arguments here. Less prone to error than
    # in every training script.

    add_experiment_args(parser)
    add_data_args(parser)
    add_environment_args(parser)
    add_model_args(parser)
    add_rl_args(parser)
    add_tracking_args(parser)
    add_oracle_args(parser)
    add_tractometer_args(parser)
