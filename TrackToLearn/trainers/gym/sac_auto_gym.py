#!/usr/bin/env python
import argparse
import json
import torch

from argparse import RawTextHelpFormatter
from os.path import join as pjoin

from TrackToLearn.algorithms.sac_auto import SACAuto
from TrackToLearn.experiment.experiment import (
    add_experiment_args,
    add_model_args)
from TrackToLearn.experiment.train import (
    add_rl_args)
from TrackToLearn.trainers.gym.gym_train import (
    GymTraining,
    add_environment_args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())


class SAC_AutoGymTraining(GymTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        # Dataset params
        path: str,
        experiment: str,
        name: str,
        env_name: str,
        # RL params
        max_ep: int,
        log_interval: int,
        action_std: float,
        lr: float,
        gamma: float,
        alpha: float,
        # Model params
        n_latent_var: int,
        hidden_layers: int,
        # Experiment params
        use_gpu: bool,
        rng_seed: int,
        render: bool,
    ):
        """
        Parameters
        ----------
        dataset_file: str
            Path to the file containing the signal data
        subject_id: str
            Subject being trained on (in the signal data)
        in_seed: str
            Path to the mask where seeds can be generated
        in_mask: str
            Path to the mask where tracking can happen
        scoring_data: str
            Path to reference streamlines that can be used for
            jumpstarting seeds
        max_ep: int
            How many episodes to run the training.
            An episode corresponds to tracking two-ways on one seed and
            training along the way
        log_interval: int
            Interval at which a valid run is done
        action_std: float
            Starting standard deviation on actions for exploration
        lr: float
            Learning rate for optimizer
        gamma: float
            Gamma parameter future reward discounting
        lmbda: float
            Lambda parameter for Generalized Advantage Estimation (GAE):
            John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan:
            “High-Dimensional Continuous Control Using Generalized
             Advantage Estimation”, 2015;
            http://arxiv.org/abs/1506.02438 arXiv:1506.02438
        K_epochs: int
            How many epochs to run the optimizer using the current samples
            SAC_Auto allows for many training runs on the same samples
        eps_clip: float
            Clipping parameter for SAC_Auto
        rng_seed: int
            Seed for general randomness
        entropy_loss_coeff: float,
            Loss coefficient on policy entropy
            Should sum to 1 with other loss coefficients
        npv: int
            How many seeds to generate per voxel
        theta: float
            Maximum angle for tracking
        min_length: int
            Minimum length for streamlines
        max_length: int
            Maximum length for streamlines
        step_size: float
            Step size for tracking
        alignment_weighting: float
            Reward coefficient for alignment with local odfs peaks
        straightness_weighting: float
            Reward coefficient for streamline straightness
        length_weighting: float
            Reward coefficient for streamline length
        target_bonus_factor: `float`
            Bonus for streamlines reaching the target mask
        exclude_penalty_factor: `float`
            Penalty for streamlines reaching the exclusion mask
        angle_penalty_factor: `float`
            Penalty for looping or too-curvy streamlines
        n_actor: int
            Batch size for tracking during valid
        n_latent_var: int
            Width of the NN layers
        add_neighborhood: float
            Use signal in neighboring voxels for model input
        # Experiment params
        use_comet: bool
            Use comet for displaying stats during training
        render: bool
            Render tracking
        run_tractometer: bool
            Run tractometer during validation to see how it's
            doing w.r.t. ground truth data
        use_gpu: bool,
            Use GPU for processing
        rng_seed: int
            Seed for general randomness
        load_teacher: str
            Path to pretrained model for imitation learning
        load_policy: str
            Path to pretrained policy
        """

        super().__init__(
            # Dataset params
            path,
            experiment,
            name,
            env_name,
            # SAC_Auto params
            max_ep,
            log_interval,
            action_std,
            lr,
            gamma,
            # Model params
            n_latent_var,
            hidden_layers,
            # Experiment params
            use_gpu,
            rng_seed,
            render,
        )

        self.alpha = alpha

    def save_hyperparameters(self):
        self.hyperparameters = {
            # RL parameters
            'id': self.name,
            'experiment': self.experiment,
            'algorithm': 'SAC_Auto',
            'max_ep': self.max_ep,
            'log_interval': self.log_interval,
            'action_std': self.action_std,
            'lr': self.lr,
            'gamma': self.gamma,
            # Data parameters
            'input_size': self.input_size,
            # Model parameters
            'experiment_path': self.experiment_path,
            'use_gpu': self.use_gpu,
            'hidden_size': self.n_latent_var,
            'hidden_layers': self.hidden_layers,
            'last_episode': self.last_episode,
        }

        directory = pjoin(self.experiment_path, "model")
        with open(
            pjoin(directory, "hyperparameters.json"),
            'w'
        ) as json_file:
            json_file.write(
                json.dumps(
                    self.hyperparameters,
                    indent=4,
                    separators=(',', ': ')))

    def get_alg(self):
        # The RL training algorithm
        alg = SACAuto(
            self.input_size,
            self.action_size,
            self.n_latent_var,
            self.hidden_layers,
            self.lr,
            self.gamma,
            self.alpha,
            1,
            False,
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

    add_environment_args(parser)
    add_model_args(parser)
    add_rl_args(parser)
    add_sac_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)

    # Finally, get experiments, and train your models:
    trpo_experiment = SAC_AutoGymTraining(
        # Dataset params
        args.path,
        args.experiment,
        args.name,
        args.env_name,
        # RL params
        args.max_ep,
        args.log_interval,
        args.alpha,
        # RL Params
        args.lr,
        args.gamma,
        args.alpha,
        # Model params
        args.n_latent_var,
        args.hidden_layers,
        # Experiment params
        args.use_gpu,
        args.rng_seed,
        args.render,
    )
    trpo_experiment.run()


if __name__ == '__main__':
    main()
