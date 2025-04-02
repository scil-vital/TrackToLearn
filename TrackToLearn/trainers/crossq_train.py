#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from argparse import RawTextHelpFormatter

import comet_ml  # noqa: F401 ugh
import torch
from comet_ml import Experiment as CometExperiment
from comet_ml import OfflineExperiment as CometOfflineExperiment

from TrackToLearn.algorithms.crossq import CrossQ
from TrackToLearn.trainers.train import (TrackToLearnTraining,
                                         add_training_args)
from TrackToLearn.utils.torch_utils import get_device
device = get_device()


class CrossQTrackToLearnTraining(TrackToLearnTraining):
    """
    Train a RL tracking agent using SAC with automatic entropy adjustment.
    """

    def __init__(
        self,
        crossq_train_dto: dict,
        comet_experiment: CometExperiment,
    ):
        """
        Parameters
        ----------
        crossq_train_dto: dict
        CrossQ training parameters
        comet_experiment: CometExperiment
        Allows for logging and experiment management.
        """

        super().__init__(
            crossq_train_dto,
            comet_experiment,
        )

        # CrossQ-specific parameters
        self.alpha = crossq_train_dto['alpha']
        self.batch_size = crossq_train_dto['batch_size']
        self.replay_size = crossq_train_dto['replay_size']

    def save_hyperparameters(self):
        """ Add CrossQ-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'CrossQ',
             'alpha': self.alpha,
             'batch_size': self.batch_size,
             'replay_size': self.replay_size})

        super().save_hyperparameters()

    def get_alg(self, max_nb_steps: int):
        alg = CrossQ(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.lr,
            self.gamma,
            self.alpha,
            self.n_actor,
            self.batch_size,
            self.replay_size,
            self.rng,
            device)
        return alg


def add_crossq_args(parser):
    parser.add_argument('--alpha', default=0.2, type=float,
                        help='Initial temperature parameter')
    parser.add_argument('--batch_size', default=2**12, type=int,
                        help='How many tuples to sample from the replay '
                        'buffer.')
    parser.add_argument('--replay_size', default=1e6, type=int,
                        help='How many tuples to store in the replay buffer.')


def parse_args():
    """ Generate a tractogram from a trained model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)
    add_training_args(parser)
    add_crossq_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)

    offline = args.comet_offline_dir is not None

    # Create comet-ml experiment
    if offline:
        experiment = CometOfflineExperiment(project_name=args.experiment,
                                            workspace=args.workspace, parse_args=False,
                                            auto_metric_logging=False,
                                            disabled=not args.use_comet,
                                            offline_directory=args.comet_offline_dir)
    else:
        experiment = CometExperiment(project_name=args.experiment,
                                     workspace=args.workspace, parse_args=False,
                                     auto_metric_logging=False,
                                     disabled=not args.use_comet)

    # Create and run experiment
    crossq_experiment = CrossQTrackToLearnTraining(
        # Dataset params
        vars(args),
        experiment
    )
    crossq_experiment.run()


if __name__ == '__main__':
    main()
