#!/usr/bin/env python
import argparse
import comet_ml  # noqa: F401 ugh
import torch

from argparse import RawTextHelpFormatter
from comet_ml import Experiment as CometExperiment

from TrackToLearn.algorithms.td3 import TD3
from TrackToLearn.experiment.experiment import (
    add_data_args,
    add_environment_args,
    add_experiment_args,
    add_model_args,
    add_tracking_args)
from TrackToLearn.experiment.train import (
    add_rl_args,
    TrackToLearnTraining)
from TrackToLearn.utils.torch_utils import get_device, assert_accelerator

device = get_device()
assert_accelerator()


class TD3TrackToLearnTraining(TrackToLearnTraining):
    """ WARNING: TD3 is no longer supported. No support will be provied.
    The code is left as example and for legacy purposes.

    Train a RL tracking agent using TD3.
    """

    def __init__(
        self,
        td3_train_dto: dict,
        comet_experiment: CometExperiment,
    ):
        """
        Parameters
        ----------
        td3_train_dto: dict
            TD3 training parameters
        comet_experiment: CometExperiment
            Allows for logging and experiment management.
        """

        super().__init__(
            td3_train_dto,
            comet_experiment,
        )

        # TD3-specific parameters
        self.action_std = td3_train_dto['action_std']
        self.batch_size = td3_train_dto['batch_size']
        self.replay_size = td3_train_dto['replay_size']

    def save_hyperparameters(self):
        """ Add TD3-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'TD3',
             'action_std': self.action_std,
             'batch_size': self.batch_size,
             'replay_size': self.replay_size})

        super().save_hyperparameters()

    def get_alg(self, max_nb_steps: int):
        alg = TD3(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.action_std,
            self.lr,
            self.gamma,
            self.n_actor,
            self.batch_size,
            self.replay_size,
            self.rng,
            device)
        return alg


def add_td3_args(parser):
    parser.add_argument('--action_std', default=0.3, type=float,
                        help='Action STD')
    parser.add_argument('--batch_size', default=2**12, type=int,
                        help='How many tuples to sample from the replay '
                             'buffer.')
    parser.add_argument('--replay_size', default=1e6, type=int,
                        help='How many tuples to store in the replay buffer.')


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

    add_td3_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)
    raise DeprecationWarning('Training with TD3 is deprecated. Please train '
                             'using SAC Auto instead.')

    experiment = CometExperiment(project_name=args.experiment,
                                 workspace=args.workspace, parse_args=False,
                                 auto_metric_logging=False,
                                 disabled=not args.use_comet)

    td3_experiment = TD3TrackToLearnTraining(
        vars(args),
        experiment
    )
    td3_experiment.run()


if __name__ == '__main__':
    main()
