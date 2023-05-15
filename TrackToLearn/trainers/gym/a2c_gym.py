#!/usr/bin/env python
import argparse
import torch

from argparse import RawTextHelpFormatter

from TrackToLearn.trainers.a2c_train import add_a2c_args
from TrackToLearn.algorithms.a2c import A2C
from TrackToLearn.experiment.experiment import (
    add_experiment_args,
    add_model_args)
from TrackToLearn.experiment.train import (
    add_rl_args)
from TrackToLearn.trainers.gym.gym_train import (
    GymTraining,
    add_environment_args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available()


class A2CGymTraining(GymTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        a2c_train_dto: dict,
    ):
        """
        Parameters
        ----------
        a2c_train_dto: dict
            A2C training parameters
        """

        super().__init__(
            a2c_train_dto,
        )

        # A2C-specific parameters
        self.action_std = a2c_train_dto['action_std']
        self.n_update = a2c_train_dto['n_update']
        self.lmbda = a2c_train_dto['lmbda']
        self.entropy_loss_coeff = a2c_train_dto['entropy_loss_coeff']

    def save_hyperparameters(self):
        """ Add A2C-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'A2C',
             'action_std': self.action_std,
             'n_update': self.n_update,
             'lmbda': self.lmbda,
             'entropy_loss_coeff': self.entropy_loss_coeff})

        super().save_hyperparameters()

    def get_alg(self):
        # The RL training algorithm
        alg = A2C(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.action_std,
            self.lr,
            self.gamma,
            self.lmbda,
            self.entropy_loss_coeff,
            self.n_update,
            self.n_actor,
            self.rng,
            device)
        return alg


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

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)

    # Finally, get experiments, and train your models:
    a2c_experiment = A2CGymTraining(
        vars(args)
    )
    a2c_experiment.run()


if __name__ == '__main__':
    main()
