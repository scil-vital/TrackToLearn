#!/usr/bin/env python
import argparse
import comet_ml  # noqa: F401 ugh
import torch

from argparse import RawTextHelpFormatter
from comet_ml import Experiment as CometExperiment

from TrackToLearn.algorithms.dqn import DQN
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


class DQNTrackToLearnTraining(TrackToLearnTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        dqn_train_dto: dict,
        comet_experiment: CometExperiment,
    ):
        """
        Parameters
        ----------
        dqn_train_dto: dict
            DQN training parameters
        comet_experiment: CometExperiment
            Allows for logging and experiment management.
        """

        super().__init__(
            dqn_train_dto,
            comet_experiment,
        )

        # DQN-specific parameters
        self.epsilon_decay = dqn_train_dto['epsilon_decay']

    def save_hyperparameters(self):
        """ Add DQN-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'DQN',
             'epsilon_decay': self.epsilon_decay})

        super().save_hyperparameters()

    def get_alg(self, max_nb_steps: int):
        alg = DQN(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.lr,
            self.gamma,
            self.epsilon_decay,
            self.n_actor,
            self.rng,
            device)
        return alg


def add_dqn_args(parser):
    parser.add_argument('--epsilon_decay', default=0.9995, type=float,
                        help='Decay parameter for exploration.')


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

    add_dqn_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)

    experiment = CometExperiment(project_name=args.experiment,
                                 workspace=args.workspace, parse_args=False,
                                 auto_metric_logging=False,
                                 disabled=not args.use_comet)

    dqn_experiment = DQNTrackToLearnTraining(
        # Dataset params
        vars(args),
        experiment
    )
    dqn_experiment.run()


if __name__ == '__main__':
    main()
