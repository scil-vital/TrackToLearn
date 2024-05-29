#!/usr/bin/env python
import argparse
from argparse import RawTextHelpFormatter

import comet_ml  # noqa: F401 ugh
import torch
from comet_ml import Experiment as CometExperiment

from TrackToLearn.algorithms.ddpg import DDPG
from TrackToLearn.experiment.train import (
    add_training_args, TrackToLearnTraining)
from TrackToLearn.utils.torch_utils import get_device, assert_accelerator

device = get_device()
assert_accelerator()


class DDPGTrackToLearnTraining(TrackToLearnTraining):
    """ WARNING: DDPG is no longer supported. No support will be provied.
    The code is left as example and for legacy purposes.

    Train a RL tracking agent using DDPG.
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
        self.batch_size = ddpg_train_dto['batch_size']
        self.replay_size = ddpg_train_dto['replay_size']

    def save_hyperparameters(self):
        """ Add DDPG-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'DDPG',
             'action_std': self.action_std,
             'batch_size': self.batch_size,
             'replay_size': self.replay_size})

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
            self.batch_size,
            self.replay_size,
            self.rng,
            device)
        return alg


def add_ddpg_args(parser):
    parser.add_argument('--action_std', default=0.35, type=float,
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

    add_training_args(parser)
    add_ddpg_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """

    raise DeprecationWarning('Training with DDPG is deprecated. Please train '
                             'using SAC Auto instead.')
    args = parse_args()
    print(args)

    experiment = CometExperiment(project_name=args.experiment,
                                 workspace=args.workspace, parse_args=False,
                                 auto_metric_logging=False,
                                 disabled=not args.use_comet)

    ddpg_experiment = DDPGTrackToLearnTraining(
        vars(args),
        experiment
    )
    ddpg_experiment.run()


if __name__ == '__main__':
    main()
