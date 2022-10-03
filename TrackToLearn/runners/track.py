#!/usr/bin/env python
import argparse
import json
import nibabel as nib
import numpy as np
import random
import torch

from argparse import RawTextHelpFormatter

from TrackToLearn.algorithms.a2c import A2C
from TrackToLearn.algorithms.acer import ACER
from TrackToLearn.algorithms.acktr import ACKTR
from TrackToLearn.algorithms.ddpg import DDPG
from TrackToLearn.algorithms.ppo import PPO
from TrackToLearn.algorithms.trpo import TRPO
from TrackToLearn.algorithms.td3 import TD3
from TrackToLearn.algorithms.sac import SAC
from TrackToLearn.algorithms.sac_auto import SACAuto
from TrackToLearn.algorithms.vpg import VPG
from TrackToLearn.datasets.utils import MRIDataVolume
from TrackToLearn.runners.experiment import (
    add_environment_args,
    add_experiment_args,
    add_tracking_args)
from TrackToLearn.runners.test import TrackToLearnTest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())


class TrackToLearnTrack(TrackToLearnTest):
    """ TrackToLearn testing script. Should work on any model trained with a
    TrackToLearn experiment
    """

    def __init__(
        self,
        # Dataset params
        path: str,
        experiment: str,
        name: str,
        signal_file: str,
        peaks_file: str,
        seeding_file: str,
        tracking_file: str,
        target_file: str,
        include_file: str,
        exclude_file: str,
        subject_id: str,
        reference_file: str,
        out_tractogram: str,
        valid_noise: float,
        policy: str,
        hyperparameters: str,
        tracking_batch_size: int,
        remove_invalid_streamlines: bool,
        step_size: float,
        n_seeds_per_voxel: int,
        min_length: float,
        max_length: float,
        interface_seeding: bool,
        fa_map_file: str = None,
    ):
        """
        """
        random_seed = 1337
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.rng = np.random.RandomState(seed=random_seed)
        random.seed(random_seed)

        self.experiment_path = path
        self.experiment = experiment
        self.name = name
        self.render = False

        self.signal_file = signal_file
        self.peaks_file = peaks_file
        self.seeding_file = seeding_file
        self.tracking_file = tracking_file
        self.target_file = target_file
        self.include_file = include_file
        self.exclude_file = exclude_file

        self.test_subject_id = self.subject_id = subject_id
        self.reference_file = reference_file
        self.out_tractogram = out_tractogram

        self.valid_noise = valid_noise
        self.policy = policy
        self.tracking_batch_size = tracking_batch_size
        self.step_size = step_size
        self.n_seeds_per_voxel = n_seeds_per_voxel
        self.min_length = min_length
        self.max_length = max_length
        self.compute_reward = False

        self.fa_map = None
        if fa_map_file is not None:
            fa_image = nib.load(
                fa_map_file)
            self.fa_map = MRIDataVolume(
                data=fa_image.get_fdata(),
                affine_vox2rasmm=fa_image.affine)

        with open(hyperparameters, 'r') as json_file:
            hyperparams = json.load(json_file)
            self.algorithm = hyperparams['algorithm']
            self.add_neighborhood = hyperparams['add_neighborhood']
            self.random_seed = np.random.randint(1000)
            self.max_angle = hyperparams['max_angle']
            self.alignment_weighting = hyperparams['alignment_weighting']
            self.straightness_weighting = hyperparams['straightness_weighting']
            self.length_weighting = hyperparams['length_weighting']
            self.target_bonus_factor = hyperparams['target_bonus_factor']
            self.exclude_penalty_factor = hyperparams['exclude_penalty_factor']
            self.angle_penalty_factor = hyperparams['angle_penalty_factor']
            self.hidden_size = hyperparams['hidden_size']
            self.n_signal = hyperparams['n_signal']
            self.n_dirs = hyperparams['n_dirs']
            self.interface_seeding = hyperparams['interface_seeding']
            self.no_retrack = hyperparams.get('no_retrack', False)

        self.comet_experiment = None
        self.remove_invalid_streamlines = remove_invalid_streamlines

    def run(self):
        """
        Main method where the magic happens
        """
        # Instanciate environment. Actions will be fed to it and new
        # states will be returned. The environment updates the streamline
        # internally

        back_env, env = self.get_tracking_envs()

        # Get example state to define NN input size
        example_state = env.reset(0, 1)
        self.input_size = example_state.shape[1]

        algs = {'VPG': VPG,
                'A2C': A2C,
                'ACER': ACER,
                'ACKTR': ACKTR,
                'PPO': PPO,
                'TRPO': TRPO,
                'DDPG': DDPG,
                'TD3': TD3,
                'SAC': SAC,
                'SACAuto': SACAuto}

        rl_alg = algs[self.algorithm]

        # The RL training algorithm
        alg = rl_alg(
            self.input_size,
            3,
            self.hidden_size,
            interface_seeding=self.interface_seeding,
            rng=self.rng,
            device=device)

        # Load pretrained policies
        alg.teacher.load(self.policy, 'last_model_state')
        alg.policy.load(self.policy, 'last_model_state')

        # Run test
        tractogram, reward = self.test(alg, env, back_env, save_model=False)
        if self.remove_invalid_streamlines:
            tractogram = self.clean_tractogram(tractogram, env.affine_vox2mask)

        # Display stats and save tractogram
        self.display(tractogram, env, reward, 0, filename=self.out_tractogram)


def add_test_args(parser):
    parser.add_argument('signal_file',
                        help='Path to the precomputed signal file')
    parser.add_argument('peaks_file',
                        help='Path to the peaks file')
    parser.add_argument('seeding_file',
                        help='Path to the seeding file')
    parser.add_argument('tracking_file',
                        help='Path to the tracking file')
    parser.add_argument('target_file',
                        help='Path to the target file')
    parser.add_argument('include_file',
                        help='Path to the include file')
    parser.add_argument('exclude_file',
                        help='Path to the exclude file')
    parser.add_argument('subject_id',
                        help='Subject id to fetch from the dataset file')
    parser.add_argument('reference_file',
                        help='Path to binary seeding mask (.nii|.nii.gz)')
    parser.add_argument('policy',
                        help='Path to the policy')
    parser.add_argument('hyperparameters',
                        help='File containing the hyperparameters for the '
                             'experiment')
    parser.add_argument('--out_tractogram', type=str, default=None,
                        help='Output tractogram. Default will be in the ' +
                             'model\'s folder')
    parser.add_argument('--remove_invalid_streamlines', action='store_true')
    parser.add_argument('--fa_map', type=str, default=None,
                        help='FA map to influence STD for probabilistic' +
                        'tracking')


def parse_args():
    """ Generate a tractogram from a trained recurrent model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    add_experiment_args(parser)
    add_test_args(parser)
    add_environment_args(parser)
    add_tracking_args(parser)

    arguments = parser.parse_args()
    return arguments


def main(experiment):
    """ Main tracking script """
    experiment.run()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    experiment = TrackToLearnTrack(
        # Dataset params
        args.path,
        args.experiment,
        args.name,
        args.signal_file,
        args.peaks_file,
        args.seeding_file,
        args.tracking_file,
        args.target_file,
        args.include_file,
        args.exclude_file,
        args.subject_id,
        args.reference_file,
        args.out_tractogram,
        args.valid_noise,
        args.policy,
        args.hyperparameters,
        args.tracking_batch_size,
        args.remove_invalid_streamlines,
        args.step_size,
        args.n_seeds_per_voxel,
        args.min_length,
        args.max_length,
        args.interface_seeding,
        args.fa_map,
    )

    main(experiment)
