#!/usr/bin/env python
import argparse
import torch

from argparse import RawTextHelpFormatter

from TrackToLearn.algorithms.trpo import TRPO
from TrackToLearn.trainers.a2c_train import add_a2c_args
from TrackToLearn.experiment.experiment import (
    add_experiment_args,
    add_model_args)
from TrackToLearn.experiment.train import (
    add_rl_args)
from TrackToLearn.trainers.gym.gym_train import (
    GymTraining,
    add_environment_args)
from TrackToLearn.trainers.trpo_train import (
    add_trpo_args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available()


class TRPOGymTraining(GymTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        trpo_train_dto: dict,
    ):
        """
        Parameters
        ----------
        trpo_train_dto: dict
            TRPO training parameters
        """

        super().__init__(
            trpo_train_dto,
        )

        # TRPO-specific parameters
        self.action_std = trpo_train_dto['action_std']
        self.n_update = trpo_train_dto['n_update']
        self.lmbda = trpo_train_dto['lmbda']
        self.delta = trpo_train_dto['delta']
        self.max_backtracks = trpo_train_dto['max_backtracks']
        self.backtrack_coeff = trpo_train_dto['backtrack_coeff']
        self.K_epochs = trpo_train_dto['K_epochs']
        self.entropy_loss_coeff = trpo_train_dto['entropy_loss_coeff']

    def save_hyperparameters(self):
        """ Add TRPO-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'TRPO',
             'n_update': self.n_update,
             'action_std': self.action_std,
             'lmbda': self.lmbda,
             'delta': self.delta,
             'max_backtracks': self.max_backtracks,
             'backtrack_coeff': self.backtrack_coeff,
             'K_epochs': self.K_epochs,
             'entropy_loss_coeff': self.entropy_loss_coeff})

        super().save_hyperparameters()

    def get_alg(self):
        # The RL training algorithm
        alg = TRPO(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.action_std,
            self.lr,
            self.gamma,
            self.lmbda,
            self.entropy_loss_coeff,
            self.delta,
            self.max_backtracks,
            self.backtrack_coeff,
            self.n_update,
            self.K_epochs,
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
    add_trpo_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)

    # Finally, get experiments, and train your models:
    trpo_experiment = TRPOGymTraining(
        # Dataset params
        vars(args),
    )
    trpo_experiment.run()


if __name__ == '__main__':
    main()
