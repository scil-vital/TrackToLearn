#!/usr/bin/env python
import argparse
import comet_ml  # noqa: F401 ugh
import torch

from argparse import RawTextHelpFormatter
from comet_ml import Experiment as CometExperiment

from TrackToLearn.trainers.a2c_train import add_a2c_args
from TrackToLearn.algorithms.acktr import ACKTR
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


class ACKTRTrackToLearnTraining(TrackToLearnTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        acktr_train_dto: dict,
        comet_experiment: CometExperiment,
    ):
        """
        Parameters
        ----------
        acktr_train_dto: dict
            ACKTR training parameters
        comet_experiment: CometExperiment
            Allows for logging and experiment management.
        """

        super().__init__(
            acktr_train_dto,
            comet_experiment,
        )

        # ACKTR-specific parameters
        self.action_std = acktr_train_dto['action_std']
        self.lmbda = acktr_train_dto['lmbda']
        self.delta = acktr_train_dto['delta']
        self.entropy_loss_coeff = acktr_train_dto['entropy_loss_coeff']

    def save_hyperparameters(self):
        """ Add ACKTR-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'ACKTR',
             'action_std': self.action_std,
             'lmbda': self.lmbda,
             'delta': self.delta,
             'entropy_loss_coeff': self.entropy_loss_coeff})

        super().save_hyperparameters()

    def get_alg(self, max_nb_steps: int):
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
            max_nb_steps,
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
    add_data_args(parser)

    add_environment_args(parser)
    add_model_args(parser)
    add_rl_args(parser)
    add_tracking_args(parser)

    add_a2c_args(parser)
    add_actkr_args(parser)

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
    actkr_experiment = ACKTRTrackToLearnTraining(
        # Dataset params
        vars(args),
        experiment,
    )
    actkr_experiment.run()


if __name__ == '__main__':
    main()
