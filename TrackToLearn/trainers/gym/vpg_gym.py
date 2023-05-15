#!/usr/bin/env python
import argparse
import torch

from argparse import RawTextHelpFormatter

from TrackToLearn.trainers.vpg_train import add_vpg_args
from TrackToLearn.algorithms.vpg import VPG
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


class VPGGymTraining(GymTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        vpg_train_dto: dict,
    ):
        """
        Parameters
        ----------
        vpg_train_dto: dict
            VPG training parameters
        """

        super().__init__(
            vpg_train_dto
        )

        # VPG-specific parameters
        self.action_std = vpg_train_dto['action_std']
        self.n_update = vpg_train_dto['n_update']
        self.entropy_loss_coeff = vpg_train_dto['entropy_loss_coeff']

    def save_hyperparameters(self):
        """ Add VPG-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'VPG',
             'n_update': self.n_update,
             'entropy_loss_coeff': self.entropy_loss_coeff})

        super().save_hyperparameters()

    def get_alg(self):
        # The RL training algorithm
        alg = VPG(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.action_std,
            self.lr,
            self.gamma,
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

    add_vpg_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)

    # Finally, get experiments, and train your models:
    vpg_experiment = VPGGymTraining(
        vars(args),
    )
    vpg_experiment.run()


if __name__ == '__main__':
    main()
