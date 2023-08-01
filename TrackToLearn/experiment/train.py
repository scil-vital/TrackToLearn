import json
import numpy as np
import random
import os
import torch

from os.path import join as pjoin

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.algorithms.shared.utils import mean_losses
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.experiment.tracker import Tracker
from TrackToLearn.experiment.ttl import TrackToLearnExperiment
from TrackToLearn.experiment.experiment import add_reward_args
from TrackToLearn.experiment.oracle_validator import OracleValidator
from TrackToLearn.experiment.tractometer_validator import (
    TractometerValidator)

assert torch.cuda.is_available(), "Training is only possible on CUDA devices."


class TrackToLearnTraining(TrackToLearnExperiment):
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
        self.experiment_path = train_dto['path']
        self.experiment = train_dto['experiment']
        self.name = train_dto['id']

        # RL parameters
        self.max_ep = train_dto['max_ep']
        self.log_interval = train_dto['log_interval']
        self.prob = train_dto['prob']
        self.lr = train_dto['lr']
        self.gamma = train_dto['gamma']

        #  Tracking parameters
        self.add_neighborhood = train_dto['add_neighborhood']
        self.step_size = train_dto['step_size']
        self.dataset_file = train_dto['dataset_file']
        self.subject_id = train_dto['subject_id']
        self.valid_dataset_file = train_dto['valid_dataset_file']
        self.valid_subject_id = train_dto['valid_subject_id']
        self.reference_file = train_dto['reference_file']
        self.rng_seed = train_dto['rng_seed']
        self.npv = train_dto['npv']

        self.theta = train_dto['theta']
        self.epsilon = train_dto['epsilon']

        self.min_length = train_dto['min_length']
        self.max_length = train_dto['max_length']
        self.interface_seeding = train_dto['interface_seeding']
        self.cmc = train_dto['cmc']
        self.asymmetric = train_dto['asymmetric']
        self.sphere = train_dto['sphere']
        self.action_type = train_dto['action_type']

        # Reward parameters
        self.alignment_weighting = train_dto['alignment_weighting']
        self.straightness_weighting = train_dto['straightness_weighting']
        self.length_weighting = train_dto['length_weighting']
        self.target_bonus_factor = train_dto['target_bonus_factor']
        self.exclude_penalty_factor = train_dto['exclude_penalty_factor']
        self.angle_penalty_factor = train_dto['angle_penalty_factor']
        self.oracle_weighting = train_dto['oracle_weighting']
        self.coverage_weighting = train_dto['coverage_weighting']

        # Model parameters
        self.hidden_dims = train_dto['hidden_dims']
        self.load_agent = train_dto['load_agent']
        self.comet_experiment = comet_experiment
        self.render = train_dto['render']
        self.run_tractometer = train_dto['run_tractometer']
        self.run_oracle = train_dto['run_oracle']
        self.last_episode = 0
        self.n_actor = train_dto['n_actor']
        self.n_signal = train_dto['n_signal']
        self.n_dirs = train_dto['n_dirs']

        self.compute_reward = True  # Always compute reward during training
        self.fa_map = None
        self.no_retrack = train_dto['no_retrack']

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

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
            'name': self.name,
            'experiment': self.experiment,
            'max_ep': self.max_ep,
            'log_interval': self.log_interval,
            'lr': self.lr,
            'gamma': self.gamma,
            # Data parameters
            'add_neighborhood': self.add_neighborhood,
            'step_size': self.step_size,
            'random_seed': self.rng_seed,
            'dataset_file': self.dataset_file,
            'subject_id': self.subject_id,
            'n_seeds_per_voxel': self.npv,
            'max_angle': self.theta,
            'max_angular_error': self.epsilon,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'cmc': self.cmc,
            'asymmetric': self.asymmetric,
            'sphere': self.sphere,
            'action_type': self.action_type,
            # Model parameters
            'experiment_path': self.experiment_path,
            'hidden_dims': self.hidden_dims,
            'last_episode': self.last_episode,
            'n_actor': self.n_actor,
            'n_signal': self.n_signal,
            'n_dirs': self.n_dirs,
            'interface_seeding': self.interface_seeding,
            'no_retrack': self.no_retrack,
            # Reward parameters
            'alignment_weighting': self.alignment_weighting,
            'straightness_weighting': self.straightness_weighting,
            'length_weighting': self.length_weighting,
            'target_bonus_factor': self.target_bonus_factor,
            'exclude_penalty_factor': self.exclude_penalty_factor,
            'angle_penalty_factor': self.angle_penalty_factor,
            'coverage_weighting': self.coverage_weighting,
            'oracle_weighting': self.oracle_weighting,
        }

    def save_hyperparameters(self):

        self.hyperparameters.update({'input_size': self.input_size,
                                     'action_size': self.action_size,
                                     'voxel_size': str(self.voxel_size)})
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

        directory = pjoin(self.experiment_path, "model")
        if not os.path.exists(directory):
            os.makedirs(directory)
        alg.agent.save(directory, "last_model_state")

    def rl_train(
        self,
        alg: RLAlgorithm,
        env: BaseEnv,
        back_env: BaseEnv,
        valid_env: BaseEnv,
        back_valid_env: BaseEnv,
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
            back_env: BaseEnv
                The backward tracking environment. Should be more or less
                the same as the "forward" tracking environment but initalized
                with half-streamlines
        """

        # Current epoch
        i_episode = 0
        # Transition counter
        t = 0

        # Initialize Trackers, which will handle streamline generation and
        # trainnig
        train_tracker = Tracker(
            alg, env, back_env, self.n_actor, self.interface_seeding,
            self.no_retrack, compress=0.0)

        valid_tracker = Tracker(
            alg, valid_env, back_valid_env, self.n_actor,
            self.interface_seeding, self.no_retrack,
            compress=0.0)

        self.validators = []

        if self.run_tractometer:
            self.validators.append(TractometerValidator(
                self.run_tractometer, self.reference_file))
        if self.run_oracle:
            self.validators.append(OracleValidator(
                self.run_oracle, self.reference_file, self.device))

        # Run tracking before training to see what an untrained network does
        valid_tractogram, valid_reward = valid_tracker.track_and_validate()
        filename = self.save_vox_tractogram(valid_tractogram)
        scores = self.score_tractogram(filename)
        self.save_model(alg)

        # Display the results of the untrained network
        self.log(
            valid_tractogram, env, valid_reward, i_episode)
        self.comet_monitor.log_losses(scores, i_episode)

        # Main training loop
        while i_episode < self.max_ep:

            # Last episode/epoch. Was initially for resuming experiments but
            # since they take so little time I just restart them from scratch
            # Not sure what to do with this
            self.last_episode = i_episode

            # Train for an episode
            tractogram, losses, reward, reward_factors = \
                train_tracker.track_and_train()

            lengths = [len(s) for s in tractogram]
            avg_length = np.mean(lengths)  # Euclidian length
            # Keep track of how many transitions were gathered
            t += sum(lengths)
            avg_reward = reward / self.n_actor

            print(
                f"Total T: {t+1} Episode Num: {i_episode+1} "
                f"Avg len: {avg_length:.3f} Avg. reward: "
                f"{avg_reward:.3f}")

            # Update monitors
            self.train_reward_monitor.update(avg_reward)
            self.train_reward_monitor.end_epoch(i_episode)
            self.train_length_monitor.update(avg_length)
            self.train_length_monitor.end_epoch(i_episode)

            i_episode += 1

            if self.use_comet and self.comet_experiment is not None:
                mean_ep_reward_factors = mean_losses(reward_factors)
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
                valid_tractogram, valid_reward = \
                    valid_tracker.track_and_validate()
                filename = self.save_vox_tractogram(valid_tractogram)
                scores = self.score_tractogram(filename)

                # Display what the network is capable-of "now"
                self.log(
                    valid_tractogram, env, valid_reward, i_episode)
                self.comet_monitor.log_losses(scores, i_episode)
                self.save_model(alg)

        # Validation run
        valid_tractogram, valid_reward = valid_tracker.track_and_validate()
        filename = self.save_vox_tractogram(valid_tractogram)
        scores = self.score_tractogram(filename)

        # Display what the network is capable-of "now"
        self.log(
            valid_tractogram, env, valid_reward, i_episode)
        self.comet_monitor.log_losses(scores, i_episode)

        self.save_model(alg)

    def run(self):
        """
        Main method where the magic happens
        """

        # Instantiate environment. Actions will be fed to it and new
        # states will be returned. The environment updates the streamline
        # internally
        back_env, env = self.get_envs()
        back_valid_env, valid_env = self.get_valid_envs()

        # Get example state to define NN input size
        self.input_size = env.get_state_size()
        self.action_size = env.get_action_size()
        self.voxel_size = env.get_voxel_size()

        # Voxel size
        self.voxel_size = env.get_voxel_size()

        max_traj_length = env.max_nb_steps

        # The RL training algorithm
        alg = self.get_alg(max_traj_length)

        # Save hyperparameters to differentiate experiments later
        self.save_hyperparameters()

        self.setup_monitors()

        # Setup comet monitors to monitor experiment as it goes along
        if self.use_comet:
            self.setup_comet()

        # If included, load pretrained policies
        if self.load_agent:
            alg.agent.load(self.load_agent)
            alg.target.load(self.load_agent)

        # Start training !
        self.rl_train(alg, env, back_env, valid_env, back_valid_env)

        torch.cuda.empty_cache()


def add_rl_args(parser):
    parser.add_argument('--max_ep', default=200000, type=int,
                        help='Number of episodes to run the training '
                        'algorithm')
    parser.add_argument('--log_interval', default=20, type=int,
                        help='Log statistics, update comet, save the model '
                        'and hyperparameters at n steps')
    parser.add_argument('--lr', default=1e-6, type=float,
                        help='Learning rate')
    parser.add_argument('--gamma', default=0.925, type=float,
                        help='Gamma param for reward discounting')

    add_reward_args(parser)
