#!/usr/bin/env python
import argparse
import comet_ml  # noqa: F401 ugh
import torch

from argparse import RawTextHelpFormatter
from comet_ml import Experiment as CometExperiment

from TrackToLearn.trainers.a2c_train import add_a2c_args
from TrackToLearn.algorithms.trpo import TRPO
from TrackToLearn.experiment.experiment import (
    add_data_args,
    add_environment_args,
    add_experiment_args,
    add_model_args,
    add_tracking_args)
from TrackToLearn.experiment.train import (
    add_rl_args,
    TrackToLearnTraining)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available()


class TRPOTrackToLearnTraining(TrackToLearnTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        trpo_train_dto: dict,
        comet_experiment: CometExperiment,
    ):
        """
        Parameters
        ----------
        trpo_train_dto: dict
            TRPO training parameters
        comet_experiment: CometExperiment
            Allows for logging and experiment management.
        """

        super().__init__(
            trpo_train_dto,
            comet_experiment,
        )

        # TRPO-specific parameters
        self.action_std = trpo_train_dto['action_std']
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
             'action_std': self.action_std,
             'lmbda': self.lmbda,
             'delta': self.delta,
             'max_backtracks': self.max_backtracks,
             'backtrack_coeff': self.backtrack_coeff,
             'K_epochs': self.K_epochs,
             'entropy_loss_coeff': self.entropy_loss_coeff})

        super().save_hyperparameters()

    def get_alg(self, max_nb_steps: int):
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
            self.K_epochs,
            max_nb_steps,
            self.n_actor,
            self.rng,
            device)
        return alg


def add_trpo_args(parser):
    parser.add_argument('--max_backtracks', default=10, type=int,
                        help='Backtracks for conjugate gradient')
    parser.add_argument('--delta', default=0.001, type=float,
                        help='Clipping parameter for TRPO')
    parser.add_argument('--backtrack_coeff', default=0.5, type=float,
                        help='Backtracking coefficient')
    parser.add_argument('--K_epochs', default=5, type=int,
                        help='Train the model for K epochs')


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
    add_trpo_args(parser)

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
    trpo_experiment = TRPOTrackToLearnTraining(
        # Dataset params
        vars(args),
        experiment,
    )
    trpo_experiment.run()


if __name__ == '__main__':
    main()
