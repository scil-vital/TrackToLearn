#!/usr/bin/env python
import argparse
import comet_ml  # noqa: F401 ugh
import torch

from argparse import RawTextHelpFormatter
from comet_ml import Experiment as CometExperiment

from TrackToLearn.algorithms.a2c import A2C
from TrackToLearn.experiment.experiment import (
    add_data_args,
    add_environment_args,
    add_experiment_args,
    add_model_args,
    add_tracking_args)
from TrackToLearn.experiment.train import (
    add_rl_args,
    TrackToLearnTraining)
from TrackToLearn.trainers.vpg_train import add_vpg_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert torch.cuda.is_available()


class A2CTrackToLearnTraining(TrackToLearnTraining):
    """
    Advantage Actor-Critic experiment.
    """

    def __init__(
        self,
        a2c_train_dto: dict,
        comet_experiment: CometExperiment,
    ):
        """
        Parameters
        ----------
        a2c_train_dto: dict
            A2C training parameters
        comet_experiment: CometExperiment
            Allows for logging and experiment management.
        """

        super().__init__(
            a2c_train_dto,
            comet_experiment,
        )

        # A2C-specific parameters
        self.action_std = a2c_train_dto['action_std']
        self.lmbda = a2c_train_dto['lmbda']
        self.entropy_loss_coeff = a2c_train_dto['entropy_loss_coeff']

    def save_hyperparameters(self):
        """ Add A2C-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'A2C',
             'action_std': self.action_std,
             'lmbda': self.lmbda,
             'entropy_loss_coeff': self.entropy_loss_coeff})

        super().save_hyperparameters()

    def get_alg(self, max_nb_steps: int):
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
            max_nb_steps,
            self.n_actor,
            self.rng,
            device)
        return alg


def add_a2c_args(parser):
    add_vpg_args(parser)
    parser.add_argument('--lmbda', default=0.95, type=float,
                        help='Lambda param for advantage discounting')


def parse_args():
    """ Generate a tractogram from a trained recurrent model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    add_experiment_args(parser)
    add_data_args(parser)

    add_environment_args(parser)
    add_model_args(parser)
    add_rl_args(parser)
    add_tracking_args(parser)

    add_a2c_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)

    experiment = CometExperiment(project_name=args.experiment,
                                 workspace='TrackToLearn', parse_args=False,
                                 auto_metric_logging=False,
                                 disabled=not args.use_comet)

    # Finally, get experiments, and train your models:
    a2c_experiment = A2CTrackToLearnTraining(
        vars(args),
        experiment,
    )
    a2c_experiment.run()


if __name__ == '__main__':
    main()
