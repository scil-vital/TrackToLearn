#!/usr/bin/env python
import argparse
import comet_ml  # noqa: F401 ugh
import torch

from argparse import RawTextHelpFormatter
from comet_ml import Experiment as CometExperiment

from TrackToLearn.algorithms.vpg import VPG
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


class VPGTrackToLearnTraining(TrackToLearnTraining):
    """
    Vanilla Policy Gradient experiment.
    """

    def __init__(
        self,
        vpg_train_dto: dict,
        comet_experiment: CometExperiment,
    ):
        """
        Parameters
        ----------
        vpg_train_dto: dict
            VPG training parameters
        comet_experiment: CometExperiment
            Allows for logging and experiment management.
        """

        super().__init__(
            vpg_train_dto,
            comet_experiment,
        )

        # VPG-specific parameters
        self.action_std = vpg_train_dto['action_std']
        self.entropy_loss_coeff = vpg_train_dto['entropy_loss_coeff']

    def save_hyperparameters(self):
        """ Add VPG-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'VPG',
             'action_std': self.action_std,
             'entropy_loss_coeff': self.entropy_loss_coeff})

        super().save_hyperparameters()

    def get_alg(self, max_nb_steps: int):
        # The RL training algorithm
        alg = VPG(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.action_std,
            self.lr,
            self.gamma,
            self.entropy_loss_coeff,
            max_nb_steps,
            self.n_actor,
            self.rng,
            device)
        return alg


def add_vpg_args(parser):
    parser.add_argument('--entropy_loss_coeff', default=0.0001, type=float,
                        help='Entropy bonus coefficient')
    parser.add_argument('--action_std', default=0.0, type=float,
                        help='Standard deviation used of the action')


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
    add_vpg_args(parser)
    add_tracking_args(parser)

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
    vpg_experiment = VPGTrackToLearnTraining(
        # Dataset params
        vars(args),
        experiment,
    )
    vpg_experiment.run()


if __name__ == '__main__':
    main()
