import json
import numpy as np
import random
import os
import torch

from os.path import join as pjoin

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.trainers.gym.gym_exp import GymExperiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available()


class GymTraining(GymExperiment):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        # Dataset params
        train_dto: dict,
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
        self.id = train_dto['id']
        self.env_name = train_dto['env_name']

        # RL parameters
        self.max_ep = train_dto['max_ep']
        self.log_interval = train_dto['log_interval']
        self.lr = train_dto['lr']
        self.gamma = train_dto['gamma']

        #  Tracking parameters
        self.rng_seed = train_dto['rng_seed']
        self.n_actor = train_dto['n_actor']

        # Model parameters
        self.use_gpu = train_dto['use_gpu']
        self.hidden_dims = train_dto['hidden_dims']
        self.render = train_dto['render']
        self.last_episode = 0

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
            'id': self.id,
            'experiment': self.experiment,
            'max_ep': self.max_ep,
            'log_interval': self.log_interval,
            'lr': self.lr,
            'gamma': self.gamma,
            # Data parameters
            # Model parameters
            'experiment_path': self.experiment_path,
            'use_gpu': self.use_gpu,
            'hidden_dims': self.hidden_dims,
            'last_episode': self.last_episode,
        }

    def save_hyperparameters(self):

        self.hyperparameters.update({'input_size': self.input_size,
                                     'action_size': self.action_size})
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
        # Run the valid before training to see what an untrained network does
        valid_reward = self.valid(
            alg, env)

        # Display the results of the untrained network
        self.display(env, valid_reward/self.n_actor, 0)

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
            losses, reward, episode_length = \
                alg.gym_train(env)

            reward /= self.n_actor

            # Keep track of how many transitions were gathered
            t += episode_length

            print(
                f"Total T: {t+1} Episode Num: {i_episode+1} "
                f"Episode T: {episode_length} Reward: {reward:.3f}")
            print(losses)

            i_episode += 1

            # Time to do a valid run and display stats
            if i_episode % self.log_interval == 0:

                # Validation run
                valid_reward = self.valid(
                    alg, env)

                # Display what the network is capable-of "now"
                self.display(
                    env,
                    valid_reward / self.n_actor,
                    i_episode)

        # Validation run
        valid_reward = self.valid(
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
        # Get example state to define NN input size
        example_state = env.reset()
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
