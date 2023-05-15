#!/usr/bin/env python
import argparse
import comet_ml  # noqa: F401 ugh
import torch

from argparse import RawTextHelpFormatter
from comet_ml import Experiment as CometExperiment

from TrackToLearn.algorithms.ddpg import DDPG
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


class DDPGTrackToLearnTraining(TrackToLearnTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        ddpg_train_dto: dict,
        comet_experiment: CometExperiment,
    ):
        """
        Parameters
        ----------
        ddpg_train_dto: dict
            DDPG training parameters
        comet_experiment: CometExperiment
            Allows for logging and experiment management.
        """

        super().__init__(
            ddpg_train_dto,
            comet_experiment,
        )

        # DDPG-specific parameters
        self.action_std = ddpg_train_dto['action_std']

    def save_hyperparameters(self):
        """ Add DDPG-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'DDPG',
             'action_std': self.action_std})

        super().save_hyperparameters()

    def get_alg(self, max_nb_steps: int):
        alg = DDPG(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.action_std,
            self.lr,
            self.gamma,
            self.n_actor,
            self.rng,
            device)
        return alg


def add_ddpg_args(parser):
    parser.add_argument('--action_std', default=0.3, type=float,
                        help='Action STD')


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

    add_ddpg_args(parser)

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

    ddpg_experiment = DDPGTrackToLearnTraining(
        vars(args),
        experiment
    )
    ddpg_experiment.run()


if __name__ == '__main__':
    main()
