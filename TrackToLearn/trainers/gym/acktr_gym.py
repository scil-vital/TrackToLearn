#!/usr/bin/env python
import argparse
import comet_ml  # noqa: F401 ugh
import torch

from argparse import RawTextHelpFormatter

from TrackToLearn.trainers.a2c_train import add_a2c_args
from TrackToLearn.algorithms.acktr import ACKTR
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


class ACKTRGymTraining(GymTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        acktr_train_dto: dict,
    ):
        """
        Parameters
        ----------
        acktr_train_dto: dict
            ACKTR training parameters
        """

        super().__init__(
            acktr_train_dto,
        )

        # ACKTR-specific parameters
        self.action_std = acktr_train_dto['action_std']
        self.n_update = acktr_train_dto['n_update']
        self.lmbda = acktr_train_dto['lmbda']
        self.delta = acktr_train_dto['delta']
        self.entropy_loss_coeff = acktr_train_dto['entropy_loss_coeff']

    def save_hyperparameters(self):
        """ Add ACKTR-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'ACKTR',
             'n_update': self.n_update,
             'action_std': self.action_std,
             'lmbda': self.lmbda,
             'delta': self.delta,
             'entropy_loss_coeff': self.entropy_loss_coeff})

        super().save_hyperparameters()

    def get_alg(self):
        # The RL training algorithm
        alg = ACKTR(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.action_std,
            self.lr,
            self.gamma,
            self.lmbda,
            self.entropy_loss_coeff,
            self.delta,
            self.n_update,
            self.n_actor,
            self.rng,
            device)
        return alg


def add_actkr_args(parser):
    parser.add_argument('--delta', default=0.001, type=float,
                        help='KL clip parameter')


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
    add_actkr_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)

    # Finally, get experiments, and train your models:
    actkr_experiment = ACKTRGymTraining(
        # Dataset params
        vars(args),
    )
    actkr_experiment.run()


if __name__ == '__main__':
    main()
