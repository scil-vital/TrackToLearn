#!/usr/bin/env python
import torch

from typing import Tuple

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.environments.env import BaseEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())


class Experiment(object):
    """ Base class for experiments
    """

    def run(self):
        """ Main method where data is loaded, classes are instanciated,
        everything is set up.
        """
        pass

    def get_envs(self) -> Tuple[BaseEnv, BaseEnv]:
        """ Build environments

        Returns:
        --------
        back_env: BaseEnv
            Backward environment that will be pre-initialized
            with half-streamlines
        env: BaseEnv
            "Forward" environment only initialized with seeds
        """
        pass

    def get_test_envs(self) -> Tuple[BaseEnv, BaseEnv]:
        """ Build environments

        Returns:
        --------
        back_env: BaseEnv
            Backward environment that will be pre-initialized
            with half-streamlines
        env: BaseEnv
            "Forward" environment only initialized with seeds
        """

        # Not sure if parameters should come from `self` of actual
        # function parameters. It feels a bit dirty to have everything
        # in `self`, but then there's a crapload of parameters

        pass

    def test(
        self,
        alg: RLAlgorithm,
        env: BaseEnv,
        back_env: BaseEnv,
        save_model: bool = True,
    ) -> Tuple[float]:
        """
        Run the tracking algorithm without noise to see how it performs

        Parameters
        ----------
        alg: RLAlgorithm
            Tracking algorithm that contains the being-trained policy
        env: BaseEnv
            Forward environment
        back_env: BaseEnv
            Backward environment
        save_model: bool
            Save the model or not

        Returns:
        --------
        tractogram: Tractogram
            validation tractogram
        reward: float
            Reward obtained during validation
        """

        pass

    def display(
        self,
        env: BaseEnv,
        valid_reward: float = 0,
        i_episode: int = 0,
        run_tractometer: bool = False,
    ):
        pass


def add_experiment_args(parser):
    parser.add_argument('path', type=str,
                        help='Path to experiment')
    parser.add_argument('experiment',
                        help='Type of experiment (usually alg + dataset). ' +
                        'Will group runs of same experiment but different ' +
                        'parameters together')
    parser.add_argument('name', type=str,
                        help='Name to experiment')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use gpu or not')
    parser.add_argument('--rng_seed', default=1337, type=int,
                        help='Seed to fix general randomness')
    parser.add_argument('--use_comet', action='store_true',
                        help='Use comet to display training or not')
    parser.add_argument('--run_tractometer', action='store_true',
                        help='Run tractometer during validation to monitor' +
                        ' how the training is doing w.r.t. ground truth')
    parser.add_argument('--render', action='store_true',
                        help='Save screenshots of tracking as it goes along.' +
                        'Preferably disabled on non-graphical environments')
    parser.add_argument('--ground_truth_folder',
                        help='Path to reference streamlines (.nii|.nii.gz)')


def add_data_args(parser):
    parser.add_argument('dataset_file',
                        help='Path to preprocessed dataset file (.hdf5)')
    parser.add_argument('subject_id',
                        help='Subject id to fetch from the dataset file')
    parser.add_argument('test_dataset_file',
                        help='Path to preprocessed dataset file (.hdf5)')
    parser.add_argument('test_subject_id',
                        help='Subject id to fetch from the dataset file')
    parser.add_argument('reference_file',
                        help='Path to binary seeding mask (.nii|.nii.gz)')


def add_offline_data_args(parser):
    parser.add_argument('dataset_file',
                        help='Path to preprocessed dataset file (.hdf5)')
    parser.add_argument('reference_file',
                        help='Path to binary seeding mask (.nii|.nii.gz)')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size')
    parser.add_argument('--precomputed_states', action='store_true',
                        help='If set, states are expected to be precomputed '
                             'in the dataset.')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Number of workers to use for data loading')


def add_environment_args(parser):
    parser.add_argument('--n_signal', default=1, type=int,
                        help='Signal at the last n positions')
    parser.add_argument('--n_dirs', default=4, type=int,
                        help='Last n steps taken')
    parser.add_argument('--add_neighborhood', default=0.75, type=float,
                        help='Add neighborhood to model input')
    parser.add_argument('--n_seeds_per_voxel', default=2, type=int,
                        help='Number of random seeds per seeding mask voxel')
    parser.add_argument('--max_angle', default=30, type=int,
                        help='Max angle for tracking')
    parser.add_argument('--min_length', default=20, type=int,
                        help='Minimum length for tracts')
    parser.add_argument('--max_length', default=200, type=int,
                        help='Maximum length for tracts')
    parser.add_argument('--alignment_weighting', default=1, type=float,
                        help='Alignment weighting for reward')
    parser.add_argument('--straightness_weighting', default=0, type=float,
                        help='Straightness weighting for reward')
    parser.add_argument('--length_weighting', default=0, type=float,
                        help='Length weighting for reward')
    parser.add_argument('--target_bonus_factor', default=0, type=float,
                        help='Bonus for streamlines reaching the target mask')
    parser.add_argument('--exclude_penalty_factor', default=0, type=float,
                        help='Penalty for streamlines reaching the exclusion '
                        'mask')
    parser.add_argument('--angle_penalty_factor', default=0, type=float,
                        help='Penalty for looping or too-curvy streamlines')
    parser.add_argument('--step_size', default=0.75, type=float,
                        help='Step size for tracking')
    parser.add_argument('--cmc', action='store_true',
                        help='If set, use Continuous Mask Criteria to stop'
                        'tracking.')
    parser.add_argument('--asymmetric', action='store_true',
                        help='If set, presume asymmetric fODFs when '
                        'computing reward.')


def add_model_args(parser):
    parser.add_argument('--recurrent', default=0, type=int,
                        help='Number of recurrent layers to use.')
    parser.add_argument('--hidden_dims', default='1024-1024', type=str,
                        help='Hidden layers of model(s).')
    parser.add_argument('--load_policy', default=None, type=str,
                        help='Path to pretrained model')


def add_tracking_args(parser):
    parser.add_argument('--tracking_batch_size', default=50000, type=int,
                        help='Number of seeds used per episode')
    parser.add_argument('--valid_noise', default=0.0, type=float,
                        help='Noise to make a probablistic output during '
                        'test/valid')
    parser.add_argument('--interface_seeding', action='store_true',
                        help='If set, don\'t track "backwards"')
    parser.add_argument('--no_retrack', action='store_true',
                        help='If set, don\'t retrack backwards')
    parser.add_argument('--stochastic', action='store_true',
                        help='If set, use the agent\'s stochastic policy '
                             'at test time.')
