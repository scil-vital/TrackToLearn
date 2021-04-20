#!/usr/bin/env python
import numpy as np
import random
import os
import torch

from comet_ml import Experiment
from os.path import join as pjoin

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.runners.experiment import TrackToLearnExperiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())


class TrackToLearnTraining(TrackToLearnExperiment):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        # Dataset params
        path: str,
        experiment: str,
        name: str,
        dataset_file: str,
        subject_id: str,
        test_dataset_file: str,
        test_subject_id: str,
        reference_file: str,
        ground_truth_folder: str,
        # RL params
        max_ep: int,
        log_interval: int,
        action_std: float,
        valid_noise: float,
        lr: float,
        gamma: float,
        # Env params
        n_seeds_per_voxel: int,
        max_angle: float,
        min_length: int,
        max_length: int,
        step_size: float,  # Step size (in mm)
        tracking_batch_size: int,
        n_signal: int,
        n_dirs: int,
        # Model params
        n_latent_var: int,
        hidden_layers: int,
        add_neighborhood: float,
        # Experiment params
        use_gpu: bool,
        rng_seed: int,
        comet_experiment: Experiment,
        render: bool,
        run_tractometer: bool,
        load_policy: str,
    ):
        """
        Parameters
        ----------
        dataset_file: str
            Path to the file containing the signal data
        subject_id: str
            Subject being trained on (in the signal data)
        seeding_file: str
            Path to the mask where seeds can be generated
        tracking_file: str
            Path to the mask where tracking can happen
        ground_truth_folder: str
            Path to reference streamlines that can be used for
            jumpstarting seeds
        target_file: str
            Path to the mask representing the tracking endpoints
        exclude_file: str
            Path to the mask reprensenting the tracking no-go zones
        max_ep: int
            How many episodes to run the training.
            An episode corresponds to tracking two-ways on one seed and
            training along the way
        log_interval: int
            Interval at which a test run is done
        action_std: float
            Starting standard deviation on actions for exploration
        lr: float
            Learning rate for optimizer
        gamma: float
            Gamma parameter future reward discounting
        n_seeds_per_voxel: int
            How many seeds to generate per voxel
        max_angle: float
            Maximum angle for tracking
        min_length: int
            Minimum length for streamlines
        max_length: int
            Maximum length for streamlines
        step_size: float
            Step size for tracking
        tracking_batch_size: int
            Batch size for tracking during test
        n_latent_var: int
            Width of the NN layers
        add_neighborhood: float
            Use signal in neighboring voxels for model input
        # Experiment params
        use_gpu: bool,
            Use GPU for processing
        rng_seed: int
            Seed for general randomness
        comet_experiment: bool
            Use comet for displaying stats during training
        render: bool
            Render tracking
        run_tractometer: bool
            Run tractometer during validation to see how it's
            doing w.r.t. ground truth data
        load_policy: str
            Path to pretrained policy
        """
        self.experiment_path = path
        self.experiment = experiment
        self.name = name

        # RL parameters
        self.max_ep = max_ep
        self.log_interval = log_interval
        self.action_std = action_std
        self.valid_noise = valid_noise
        self.lr = lr
        self.gamma = gamma

        #  Tracking parameters
        self.add_neighborhood = add_neighborhood
        self.step_size = step_size
        self.dataset_file = dataset_file
        self.subject_id = subject_id
        self.test_dataset_file = test_dataset_file
        self.test_subject_id = test_subject_id
        self.reference_file = reference_file
        self.ground_truth_folder = ground_truth_folder
        self.rng_seed = rng_seed
        self.n_seeds_per_voxel = n_seeds_per_voxel
        self.max_angle = max_angle
        self.min_length = min_length
        self.max_length = max_length

        # Model parameters
        self.use_gpu = use_gpu
        self.n_latent_var = n_latent_var
        self.hidden_layers = hidden_layers
        self.load_policy = load_policy
        self.comet_experiment = comet_experiment
        self.render = render
        self.run_tractometer = run_tractometer
        self.last_episode = 0
        self.tracking_batch_size = tracking_batch_size
        self.n_signal = n_signal
        self.n_dirs = n_dirs

        self.compute_reward = True  # Always compute reward during training
        self.fa_map = None

        # RNG
        torch.manual_seed(self.rng_seed)
        np.random.seed(self.rng_seed)
        self.rng = np.random.RandomState(seed=self.rng_seed)
        random.seed(self.rng_seed)

        directory = os.path.dirname(pjoin(self.experiment_path, "model"))
        if not os.path.exists(directory):
            os.makedirs(directory)

    def rl_train(
        self,
        alg: RLAlgorithm,
        env: BaseEnv,
        back_env: BaseEnv,
        test_env: BaseEnv,
        back_test_env: BaseEnv,
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
        # Tractogram containing all the episodes. Might be useful I guess
        # Run the test before training to see what an untrained network does
        valid_tractogram, valid_reward = self.test(
            alg, test_env, back_test_env)

        # Display the results of the untrained network
        self.display(valid_tractogram, env, valid_reward, 0)

        # Current epoch
        i_episode = 0
        # Transition counter
        t = 0
        # Transition counter for logging
        t_log = 0

        # Main training loop
        while i_episode <= self.max_ep:

            # Last episode/epoch. Was initially for resuming experiments but
            # since they take so little time I just restart them from scratch
            # Not sure what to do with this
            self.last_episode = i_episode

            # Run the episode
            _, actor_loss, critic_loss, reward, episode_length = \
                alg.run_train(env, back_env)

            # Keep track of how many transitions were gathered
            t += episode_length
            t_log += episode_length

            print(
                f"Total T: {t+1} Episode Num: {i_episode+1} "
                f"Episode T: {episode_length} Reward: {reward:.3f}")

            # Update monitors
            self.train_reward_monitor.update(reward)
            self.train_reward_monitor.end_epoch(i_episode)
            self.actor_loss_monitor.update(actor_loss)
            self.actor_loss_monitor.end_epoch(i_episode)
            self.critic_loss_monitor.update(critic_loss)
            self.critic_loss_monitor.end_epoch(i_episode)

            i_episode += 1
            if self.comet_experiment is not None:
                self.comet_monitor.update_train(
                    self.train_reward_monitor, self.actor_loss_monitor,
                    self.critic_loss_monitor, i_episode)

            # Time to do a valid run and display stats
            if t_log > self.log_interval:

                # Validation run
                valid_tractogram, valid_reward = self.test(
                    alg, test_env, back_test_env)

                # Display what the network is capable-of "now"
                self.display(
                    valid_tractogram,
                    env,
                    valid_reward,
                    i_episode,
                    self.run_tractometer)

                # Reset the logging counter
                t_log = t_log - self.log_interval  # to reduce "drift"

        # Validation run
        valid_tractogram, valid_reward = self.test(
            alg, test_env, back_test_env)

        # Display what the network is capable-of "now"
        self.display(
            valid_tractogram,
            env,
            valid_reward,
            i_episode,
            run_tractometer=True)

    def run(self):
        """
        Main method where the magic happens
        """

        # Instanciate environment. Actions will be fed to it and new
        # states will be returned. The environment updates the streamline
        # internally
        back_env, env = self.get_envs()
        back_test_env, test_env = self.get_test_envs()

        # Get example state to define NN input size
        example_state = env.reset(0, 1)
        self.input_size = example_state.shape[1]

        # The RL training algorithm
        alg = self.get_alg()

        # Save hyperparameters to differentiate experiments later
        self.save_hyperparameters()

        self.setup_monitors()

        # Setup comet monitors to monitor experiment as it goes along
        if True:
            self.setup_comet()

        # If included, load pretrained policies
        if self.load_policy:
            alg.policy.load(self.load_policy)
            alg.target.load(self.load_policy)

        # Start training !
        self.rl_train(alg, env, back_env, test_env, back_test_env)

        torch.cuda.empty_cache()


def add_rl_args(parser):
    parser.add_argument('--max_ep', default=200000, type=int,
                        help='Number of episodes to run the training '
                        'algorithm')
    parser.add_argument('--log_interval', default=20, type=int,
                        help='Log statistics, update comet, save the model '
                        'and hyperparameters at n steps')
    parser.add_argument('--action_std', default=0.4, type=float,
                        help='Standard deviation used of the action')
    parser.add_argument('--lr', default=1e-6, type=float,
                        help='Learning rate')
    parser.add_argument('--gamma', default=0.925, type=float,
                        help='Gamma param for reward discounting')
