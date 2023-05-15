#!/usr/bin/env python
import argparse
import comet_ml  # noqa: F401 ugh
import torch

from argparse import RawTextHelpFormatter
from comet_ml import Experiment as CometExperiment

from TrackToLearn.algorithms.sac import SAC
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


class SACTrackToLearnTraining(TrackToLearnTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        sac_train_dto: dict,
        comet_experiment: CometExperiment,
    ):
        """
        Parameters
        ----------
        sac_train_dto: dict
            SAC training parameters
        comet_experiment: CometExperiment
            Allows for logging and experiment management.
        """

        super().__init__(
            sac_train_dto,
            comet_experiment,
        )

        # SAC-specific parameters
        self.alpha = sac_train_dto['alpha']

    def save_hyperparameters(self):
        """ Add SAC-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'SAC',
             'alpha': self.alpha})

        super().save_hyperparameters()

    def get_alg(self, max_nb_steps: int):
        alg = SAC(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.lr,
            self.gamma,
            self.alpha,
            self.n_actor,
            self.rng,
            device)
        return alg


def add_sac_args(parser):
    parser.add_argument('--alpha', default=0.2, type=float,
                        help='Temperature parameter')


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

    add_sac_args(parser)

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

    sac_experiment = SACTrackToLearnTraining(
        # Dataset params
        vars(args),
        experiment
    )
    sac_experiment.run()


if __name__ == '__main__':
    main()
