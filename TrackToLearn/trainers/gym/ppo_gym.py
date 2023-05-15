#!/usr/bin/env python
import argparse
import torch

from argparse import RawTextHelpFormatter

from TrackToLearn.trainers.a2c_train import add_a2c_args
from TrackToLearn.algorithms.ppo import PPO
from TrackToLearn.experiment.experiment import (
    add_experiment_args,
    add_model_args)
from TrackToLearn.experiment.train import (
    add_rl_args)
from TrackToLearn.trainers.gym.gym_train import (
    GymTraining,
    add_environment_args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert (torch.cuda.is_available())


class PPOGymTraining(GymTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        ppo_train_dto: dict,
    ):
        """
        Parameters
        ----------
        ppo_train_dto: dict
            PPO training parameters
        """

        super().__init__(
            ppo_train_dto,
        )

        # PPO-specific parameters
        self.action_std = ppo_train_dto['action_std']
        self.n_update = ppo_train_dto['n_update']
        self.lmbda = ppo_train_dto['lmbda']
        self.eps_clip = ppo_train_dto['eps_clip']
        self.K_epochs = ppo_train_dto['K_epochs']
        self.entropy_loss_coeff = ppo_train_dto['entropy_loss_coeff']

    def save_hyperparameters(self):
        """ Add PPO-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'PPO',
             'n_update': self.n_update,
             'action_std': self.action_std,
             'lmbda': self.lmbda,
             'eps_clip': self.eps_clip,
             'K_epochs': self.K_epochs,
             'entropy_loss_coeff': self.entropy_loss_coeff})

        super().save_hyperparameters()

    def get_alg(self):
        # The RL training algorithm
        alg = PPO(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.action_std,
            self.lr,
            self.gamma,
            self.lmbda,
            self.K_epochs,
            self.n_update,
            self.eps_clip,
            self.entropy_loss_coeff,
            self.n_actor,
            self.rng,
            device)
        return alg


def add_ppo_args(parser):
    parser.add_argument('--eps_clip', default=0.001, type=float,
                        help='Clipping parameter for PPO')
    parser.add_argument('--K_epochs', default=1, type=int,
                        help='Train the model for K epochs')


def parse_args():
    """ Generate a tractogram from a trained recurrent model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    add_experiment_args(parser)

    add_environment_args(parser)
    add_model_args(parser)
    add_rl_args(parser)

    add_a2c_args(parser)
    add_ppo_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)

    # Finally, get experiments, and train your models:
    ppo_experiment = PPOGymTraining(
        # Dataset params
        vars(args)
    )
    ppo_experiment.run()


if __name__ == '__main__':
    main()
