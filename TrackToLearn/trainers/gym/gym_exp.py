import os
import torch

from os.path import join as pjoin

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.environments.gym.gym_env import GymWrapper
from TrackToLearn.experiment.experiment import Experiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available()


class GymExperiment(Experiment):
    """
    """

    def run(self):
        """ Main method where data is loaded, classes are instanciated,
        everything is set up.
        """
        pass

    def setup_monitors(self):
        #  RL monitors
        pass

    def setup_comet(self, prefix=''):
        """ Setup comet environment
        """
        pass

    def get_envs(self) -> BaseEnv:
        """ Build environment

        Returns:
        --------
        env: BaseEnv

        """
        kwargs = {}
        if self.render:
            kwargs.update({'render_mode': 'human'})
        env = GymWrapper(self.env_name, self.n_actor, **kwargs)
        return env

    def get_valid_envs(self) -> BaseEnv:
        """ Build environment

        Returns:
        --------
        env: BaseEnv
        """

        env = GymWrapper(self.env_name, 10)
        return env

    def valid(
        self,
        alg: RLAlgorithm,
        env: BaseEnv,
        save_model: bool = True,
    ) -> float:
        """
        Run the tracking algorithm without noise to see how it performs

        Parameters
        ----------
        alg: RLAlgorithm
            Tracking algorithm that contains the being-trained policy
        env: BaseEnv
            Forward environment
        save_model: bool
            Save the model or not

        Returns:
        --------
        reward: float
            Reward obtained during validation
        """

        # Save the model so it can be loaded by the tracking
        if save_model:

            directory = pjoin(self.experiment_path, "model")
            if not os.path.exists(directory):
                os.makedirs(directory)
            alg.policy.save(directory, "last_model_state")

        # Launch the tracking
        reward = alg.gym_validation(
            env, self.render)

        return reward

    def display(
        self,
        env: BaseEnv,
        valid_reward: float = 0,
        i_episode: int = 0,
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

        print('---------------------------------------------------')
        print(self.experiment_path)
        print('Episode {} \t total reward: {}'.format(
            i_episode,
            valid_reward))
        print('---------------------------------------------------')
