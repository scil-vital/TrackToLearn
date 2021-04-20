#!/usr/bin/env python
import argparse
import comet_ml  # noqa: F401 ugh
import json
import os
import torch

from argparse import RawTextHelpFormatter
from comet_ml import Experiment
from os.path import join as pjoin

from TrackToLearn.algorithms.td3 import TD3
from TrackToLearn.runners.experiment import (
    add_data_args,
    add_environment_args,
    add_experiment_args,
    add_model_args,
    add_tracking_args)
from TrackToLearn.runners.train import (
    add_rl_args,
    TrackToLearnTraining)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())


class TD3TrackToLearnTraining(TrackToLearnTraining):
    """
    Main RL tracking experiment
    """

    def __init__(
        self,
        # Dataset params
        path: str,
        experiment: str,
        name: str,
        dataset_file: str,
        subject_id: str,
        test_dataset_file: str,
        test_subject_id: str,
        reference_file: str,
        ground_truth_folder: str,
        # TD3 params
        max_ep: int,
        log_interval: int,
        action_std: float,
        valid_noise: float,
        lr: float,
        gamma: float,
        training_batch_size: int,
        # Env params
        n_seeds_per_voxel: int,
        max_angle: float,
        min_length: int,
        max_length: int,
        step_size: float,  # Step size (in mm)
        tracking_batch_size: int,
        n_signal: int,
        n_dirs: int,
        # Model params
        n_latent_var: int,
        hidden_layers: int,
        add_neighborhood: float,
        # Experiment params
        use_gpu: bool,
        rng_seed: int,
        comet_experiment: Experiment,
        render: bool,
        run_tractometer: bool,
        load_policy: str,
    ):
        """
        Parameters
        ----------
        dataset_file: str
            Path to the file containing the signal data
        subject_id: str
            Subject being trained on (in the signal data)
        ground_truth_folder: str
            Path to reference streamlines that can be used for
            jumpstarting seeds
        max_ep: int
            How many episodes to run the training.
            An episode corresponds to tracking two-ways on one seed and
            training along the way
        log_interval: int
            Interval at which a test run is done
        action_std: float
            Starting standard deviation on actions for exploration
        lr: float
            Learning rate for optimizer
        gamma: float
            Gamma parameter future reward discounting
        training_batch_size: int
            How many samples to use in policy update
        n_seeds_per_voxel: int
            How many seeds to generate per voxel
        max_angle: float
            Maximum angle for tracking
        min_length: int
            Minimum length for streamlines
        max_length: int
            Maximum length for streamlines
        step_size: float
            Step size for tracking
        tracking_batch_size: int
            Batch size for tracking during test
        n_latent_var: int
            Width of the NN layers
        add_neighborhood: float
            Use signal in neighboring voxels for model input
        # Experiment params
        use_gpu: bool,
            Use GPU for processing
        rng_seed: int
            Seed for general randomness
        comet_experiment: Experiment
            Use comet for displaying stats during training
        render: bool
            Render tracking (to file)
        run_tractometer: bool
            Run tractometer during validation to see how it's
            doing w.r.t. ground truth data
        load_policy: str
            Path to pretrained policy
        """

        super().__init__(
            # Dataset params
            path,
            experiment,
            name,
            dataset_file,
            subject_id,
            test_dataset_file,
            test_subject_id,
            reference_file,
            ground_truth_folder,
            # TD3 params
            max_ep,
            log_interval,
            action_std,
            valid_noise,
            lr,
            gamma,
            # Env params
            n_seeds_per_voxel,
            max_angle,
            min_length,
            max_length,
            step_size,  # Step size (in mm)
            tracking_batch_size,
            n_signal,
            n_dirs,
            # Model params
            n_latent_var,
            hidden_layers,
            add_neighborhood,
            # Experiment params
            use_gpu,
            rng_seed,
            comet_experiment,
            render,
            run_tractometer,
            load_policy
        )

        self.training_batch_size = training_batch_size

    def save_hyperparameters(self):
        self.hyperparameters = {
            # RL parameters
            'id': self.name,
            'experiment': self.experiment,
            'algorithm': 'TD3',
            'max_ep': self.max_ep,
            'log_interval': self.log_interval,
            'action_std': self.action_std,
            'lr': self.lr,
            'gamma': self.gamma,
            # Data parameters
            'input_size': self.input_size,
            'add_neighborhood': self.add_neighborhood,
            'step_size': self.step_size,
            'random_seed': self.rng_seed,
            'dataset_file': self.dataset_file,
            'subject_id': self.subject_id,
            'n_seeds_per_voxel': self.n_seeds_per_voxel,
            'max_angle': self.max_angle,
            'min_length': self.min_length,
            'max_length': self.max_length,
            # Model parameters
            'experiment_path': self.experiment_path,
            'use_gpu': self.use_gpu,
            'hidden_size': self.n_latent_var,
            'hidden_layers': self.hidden_layers,
            'last_episode': self.last_episode,
            'tracking_batch_size': self.tracking_batch_size,
            'n_signal': self.n_signal,
            'n_dirs': self.n_dirs,
            # Reward parameters
        }
        directory = os.path.dirname(pjoin(self.experiment_path, "model"))

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
        alg = TD3(
            self.input_size,
            3,
            self.n_latent_var,
            self.hidden_layers,
            self.action_std,
            self.lr,
            self.gamma,
            self.training_batch_size,
            self.rng,
            device)
        return alg


def add_td3_args(parser):
    parser.add_argument('--training_batch_size', default=2**14, type=int,
                        help='Number of seeds used per episode')


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

    experiment = Experiment(project_name=args.experiment,
                            workspace='TrackToLearn', parse_args=False,
                            auto_metric_logging=False,
                            disabled=not args.use_comet)

    td3_experiment = TD3TrackToLearnTraining(
        # Dataset params
        args.path,
        args.experiment,
        args.name,
        args.dataset_file,
        args.subject_id,
        args.test_dataset_file,
        args.test_subject_id,
        args.reference_file,
        args.ground_truth_folder,
        # RL params
        args.max_ep,
        args.log_interval,
        args.action_std,
        args.valid_noise,
        args.lr,
        args.gamma,
        # TD3
        args.training_batch_size,
        # Env params
        args.n_seeds_per_voxel,
        args.max_angle,
        args.min_length,
        args.max_length,
        args.step_size,  # Step size (in mm)
        args.tracking_batch_size,
        args.n_signal,
        args.n_dirs,
        # Model params
        args.n_latent_var,
        args.hidden_layers,
        args.add_neighborhood,
        # Experiment params
        args.use_gpu,
        args.rng_seed,
        experiment,
        args.render,
        args.run_tractometer,
        args.load_policy,
    )
    td3_experiment.run()


if __name__ == '__main__':
    main()
