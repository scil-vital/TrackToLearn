#!/usr/bin/env python
import numpy as np
import random
import os
import torch

from os.path import join as pjoin

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.runners.gym_exp import GymExperiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())


class GymTraining(GymExperiment):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        # Dataset params
        path: str,
        experiment: str,
        name: str,
        env_name: str,
        # RL params
        max_ep: int,
        log_interval: int,
        action_std: float,
        lr: float,
        gamma: float,
        # Model params
        n_latent_var: int,
        hidden_layers: int,
        # Experiment params
        use_gpu: bool,
        rng_seed: int,
        render: bool,
    ):
        """
        Parameters
        ----------
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
        n_latent_var: int
            Width of the NN layers
        add_neighborhood: float
            Use signal in neighboring voxels for model input
        # Experiment params
        use_gpu: bool,
            Use GPU for processing
        rng_seed: int
            Seed for general randomness
        render: bool
            Render tracking
        """
        self.experiment_path = path
        self.experiment = experiment
        self.name = name
        self.env_name = env_name

        # RL parameters
        self.max_ep = max_ep
        self.log_interval = log_interval
        self.action_std = action_std
        self.lr = lr
        self.gamma = gamma

        #  Tracking parameters
        self.rng_seed = rng_seed

        # Model parameters
        self.use_gpu = use_gpu
        self.n_latent_var = n_latent_var
        self.hidden_layers = hidden_layers
        self.render = render
        self.last_episode = 0

        # RNG
        torch.manual_seed(self.rng_seed)
        np.random.seed(self.rng_seed)
        self.rng = np.random.RandomState(seed=self.rng_seed)
        random.seed(self.rng_seed)

        directory = pjoin(self.experiment_path, 'model')
        if not os.path.exists(directory):
            os.makedirs(directory)

    def rl_train(
        self,
        alg: RLAlgorithm,
        env: BaseEnv,
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
        valid_reward = self.test(
            alg, env)

        # Display the results of the untrained network
        self.display(env, valid_reward, 0)

        # Current epoch
        i_episode = 0
        # Transition counter
        t = 0

        # Main training loop
        while i_episode < self.max_ep:

            # Last episode/epoch. Was initially for resuming experiments but
            # since they take so little time I just restart them from scratch
            # Not sure what to do with this
            self.last_episode = i_episode

            # Run the episode
            actor_loss, critic_loss, reward, episode_length = \
                alg.gym_train(env)

            # Keep track of how many transitions were gathered
            t += episode_length

            print(
                f"Total T: {t+1} Episode Num: {i_episode+1} "
                f"Episode T: {episode_length} Reward: {reward:.3f}")

            i_episode += 1

            # Time to do a valid run and display stats
            if i_episode % self.log_interval == 0:

                # Validation run
                valid_reward = self.test(
                    alg, env)

                # Display what the network is capable-of "now"
                self.display(
                    env,
                    valid_reward,
                    i_episode)

        # Validation run
        valid_reward = self.test(
            alg, env)

        # Display what the network is capable-of "now"
        self.display(
            env,
            valid_reward,
            i_episode)

    def run(self):
        """
        Main method where the magic happens
        """

        # Instanciate environment. Actions will be fed to it and new
        # states will be returned. The environment updates the streamline
        # internally
        env = self.get_envs()
        env.render()
        # Get example state to define NN input size
        example_state = env.reset()
        env.render(close=True)
        self.input_size = example_state.shape[1]
        self.n_trajectories = example_state.shape[0]
        self.action_size = env._inner_envs[0].action_space.shape[0]

        # The RL training algorithm
        alg = self.get_alg()

        # Save hyperparameters to differentiate experiments later
        self.save_hyperparameters()

        # Start training !
        self.rl_train(alg, env)

        torch.cuda.empty_cache()


def add_environment_args(parser):
    parser.add_argument('env_name', type=str,
                        help='Gym env name')
