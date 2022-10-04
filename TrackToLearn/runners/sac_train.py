#!/usr/bin/env python
import argparse
import comet_ml  # noqa: F401 ugh
import json
import torch

from argparse import RawTextHelpFormatter
from comet_ml import Experiment
from os.path import join as pjoin

from TrackToLearn.algorithms.sac import SAC
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


class SACTrackToLearnTraining(TrackToLearnTraining):
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
        # SAC params
        max_ep: int,
        log_interval: int,
        valid_noise: float,
        lr: float,
        gamma: float,
        alpha: float,
        training_batch_size: int,
        # Env params
        n_seeds_per_voxel: int,
        max_angle: float,
        min_length: int,
        max_length: int,
        step_size: float,  # Step size (in mm)
        alignment_weighting: float,
        straightness_weighting: float,
        length_weighting: float,
        target_bonus_factor: float,
        exclude_penalty_factor: float,
        angle_penalty_factor: float,
        tracking_batch_size: int,
        n_signal: int,
        n_dirs: int,
        interface_seeding: bool,
        no_retrack: bool,
        # Model params
        hidden_dims: str,
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
        seeding_file: str
            Path to the mask where seeds can be generated
        tracking_file: str
            Path to the mask where tracking can happen
        ground_truth_folder: str
            Path to reference streamlines that can be used for
            jumpstarting seeds
        target_file: str
            Path to the mask representing the tracking endpoints
        exclude_file: str
            Path to the mask reprensenting the tracking no-go zones
        max_ep: int
            How many episodes to run the training.
            An episode corresponds to tracking two-ways on one seed and
            training along the way
        log_interval: int
            Interval at which a test run is done
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

        self.training_batch_size = training_batch_size

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
            # SAC params
            max_ep,
            log_interval,
            alpha,
            valid_noise,
            lr,
            gamma,
            # Env params
            n_seeds_per_voxel,
            max_angle,
            min_length,
            max_length,
            step_size,  # Step size (in mm)
            alignment_weighting,
            straightness_weighting,
            length_weighting,
            target_bonus_factor,
            exclude_penalty_factor,
            angle_penalty_factor,
            tracking_batch_size,
            n_signal,
            n_dirs,
            interface_seeding,
            no_retrack,
            # Model params
            hidden_dims,
            add_neighborhood,
            # Experiment params
            use_gpu,
            rng_seed,
            comet_experiment,
            render,
            run_tractometer,
            load_policy
        )

        self.alpha = alpha

    def save_hyperparameters(self):
        self.hyperparameters = {
            # RL parameters
            'id': self.name,
            'experiment': self.experiment,
            'algorithm': 'SAC',
            'max_ep': self.max_ep,
            'log_interval': self.log_interval,
            'lr': self.lr,
            'gamma': self.gamma,
            'alpha': self.alpha,
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
            'hidden_dims': self.hidden_dims,
            'last_episode': self.last_episode,
            'tracking_batch_size': self.tracking_batch_size,
            'n_signal': self.n_signal,
            'n_dirs': self.n_dirs,
            'interface_seeding': self.interface_seeding,
            'no_retrack': self.no_retrack,
            # Reward parameters
            'alignment_weighting': self.alignment_weighting,
            'straightness_weighting': self.straightness_weighting,
            'length_weighting': self.length_weighting,
            'target_bonus_factor': self.target_bonus_factor,
            'exclude_penalty_factor': self.exclude_penalty_factor,
            'angle_penalty_factor': self.angle_penalty_factor
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
        alg = SAC(
            self.input_size,
            3,
            self.hidden_dims,
            self.lr,
            self.gamma,
            self.alpha,
            self.training_batch_size,
            self.interface_seeding,
            self.rng,
            device)
        return alg


def add_sac_args(parser):
    parser.add_argument('--alpha', default=0.2, type=float,
                        help='Temperature parameter')
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

    add_sac_args(parser)

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

    sac_experiment = SACTrackToLearnTraining(
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
        args.valid_noise,
        args.lr,
        args.gamma,
        args.alpha,
        # SAC
        args.training_batch_size,
        # Env params
        args.n_seeds_per_voxel,
        args.max_angle,
        args.min_length,
        args.max_length,
        args.step_size,  # Step size (in mm)
        args.alignment_weighting,
        args.straightness_weighting,
        args.length_weighting,
        args.target_bonus_factor,
        args.exclude_penalty_factor,
        args.angle_penalty_factor,
        args.tracking_batch_size,
        args.n_signal,
        args.n_dirs,
        args.interface_seeding,
        args.no_retrack,
        # Model params
        args.hidden_dims,
        args.add_neighborhood,
        # Experiment params
        args.use_gpu,
        args.rng_seed,
        experiment,
        args.render,
        args.run_tractometer,
        args.load_policy,
    )
    sac_experiment.run()


if __name__ == '__main__':
    main()
