#!/usr/bin/env python
import argparse
import comet_ml  # noqa: F401 ugh
import torch

from argparse import RawTextHelpFormatter
from comet_ml import Experiment as CometExperiment

from TrackToLearn.trainers.a2c_train import add_a2c_args
from TrackToLearn.algorithms.ppo import PPO
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


class PPOTrackToLearnTraining(TrackToLearnTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        ppo_train_dto: dict,
        comet_experiment: CometExperiment,
    ):
        """
        Parameters
        ----------
        ppo_train_dto: dict
            PPO training parameters
        comet_experiment: CometExperiment
            Allows for logging and experiment management.
        """

        super().__init__(
            ppo_train_dto,
            comet_experiment,
        )

        # PPO-specific parameters
        self.action_std = ppo_train_dto['action_std']
        self.lmbda = ppo_train_dto['lmbda']
        self.eps_clip = ppo_train_dto['eps_clip']
        self.K_epochs = ppo_train_dto['K_epochs']
        self.entropy_loss_coeff = ppo_train_dto['entropy_loss_coeff']

    def save_hyperparameters(self):
        """ Add PPO-specific hyperparameters to self.hyperparameters
        then save to file.
        """

        self.hyperparameters.update(
            {'algorithm': 'PPO',
             'action_std': self.action_std,
             'lmbda': self.lmbda,
             'eps_clip': self.eps_clip,
             'K_epochs': self.K_epochs,
             'entropy_loss_coeff': self.entropy_loss_coeff})

        super().save_hyperparameters()

    def get_alg(self, max_nb_steps: int):
        # The RL training algorithm
        alg = PPO(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            self.action_std,
            self.lr,
            self.gamma,
            self.lmbda,
            self.K_epochs,
            self.eps_clip,
            self.entropy_loss_coeff,
            max_nb_steps,
            self.n_actor,
            self.rng,
            device)
        return alg


def add_ppo_args(parser):
    parser.add_argument('--K_epochs', default=50, type=int,
                        help='Train the model for K epochs')
    parser.add_argument('--eps_clip', default=0.2, type=float,
                        help='Clipping parameter for PPO')


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
    add_ppo_args(parser)

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
    ppo_experiment = PPOTrackToLearnTraining(
        # Dataset params
        vars(args),
        experiment
    )
    ppo_experiment.run()


if __name__ == '__main__':
    main()
