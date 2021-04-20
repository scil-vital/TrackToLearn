#!/usr/bin/env python
import argparse
import json
import nibabel as nib
import numpy as np
import random
import torch

from argparse import RawTextHelpFormatter
from dipy.tracking.metrics import length as slength, winding
from nibabel.streamlines import Tractogram

from TrackToLearn.algorithms.ddpg import DDPG
from TrackToLearn.algorithms.ppo import PPO
from TrackToLearn.algorithms.td3 import TD3
from TrackToLearn.algorithms.sac import SAC
from TrackToLearn.algorithms.sac_auto import SACAuto
from TrackToLearn.algorithms.vpg import VPG
from TrackToLearn.datasets.utils import MRIDataVolume
from TrackToLearn.runners.experiment import (
    add_environment_args,
    add_experiment_args,
    add_tracking_args,
    TrackToLearnExperiment)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())


class TrackToLearnTest(TrackToLearnExperiment):
    """ TrackToLearn testing script. Should work on any model trained with a TrackToLearn
    experiment, RL or not
    """

    def __init__(
        self,
        # Dataset params
        path: str,
        experiment: str,
        name: str,
        dataset_file: str,
        subject_id: str,
        reference_file: str,
        ground_truth_folder: str,
        valid_noise: float,
        policy: str,
        stochastic: bool,
        hyperparameters: str,
        tracking_batch_size: int,
        remove_invalid_streamlines: bool,
        step_size: float,
        n_seeds_per_voxel: int,
        min_length: float,
        max_length: float,
        gm_seeding: bool,
        fa_map_file: str = None,
    ):
        """
        """
        self.experiment_path = path
        self.experiment = experiment
        self.name = name
        self.render = False

        self.test_dataset_file = self.dataset_file = dataset_file
        self.test_subject_id = self.subject_id = subject_id
        self.reference_file = reference_file
        self.ground_truth_folder = ground_truth_folder
        self.valid_noise = valid_noise
        self.policy = policy
        self.tracking_batch_size = tracking_batch_size
        self.step_size = step_size
        self.n_seeds_per_voxel = n_seeds_per_voxel
        self.min_length = min_length
        self.max_length = max_length
        self.compute_reward = False
        self.stochastic = stochastic

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
            self.gm_seeding = hyperparams['gm_seeding']

        self.comet_experiment = None
        self.remove_invalid_streamlines = remove_invalid_streamlines

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.rng = np.random.RandomState(seed=self.random_seed)
        random.seed(self.random_seed)

    def clean_tractogram(self, tractogram, affine_vox2mask):
        """
        Remove potential "non-connections" by filtering according to
        curvature, length and mask

        Parameters:
        -----------
        tractogram: Tractogram
            Full tractogram

        Returns:
        --------
        tractogram: Tractogram
            Filtered tractogram
        """
        print('Cleaning tractogram ... ', end='', flush=True)

        streamlines = tractogram.streamlines

        # # Filter by curvature
        # dirty_mask = is_flag_set(
        #     stopping_flags, StoppingFlags.STOPPING_CURVATURE)
        dirty_mask = np.zeros(len(streamlines))

        # Filter by length unless the streamline ends in GM
        # Example case: Bundle 3 of fibercup tends to be shorter than 35

        lengths = [slength(s) for s in streamlines]
        short_lengths = np.asarray(
            [lgt <= self.min_length for lgt in lengths])

        dirty_mask = np.logical_or(short_lengths, dirty_mask)

        long_lengths = np.asarray(
            [lgt > 200. for lgt in lengths])

        dirty_mask = np.logical_or(long_lengths, dirty_mask)

        # start_mask = is_inside_mask(
        #     np.asarray([s[0] for s in streamlines])[:, None],
        #     self.target_mask.data, affine_vox2mask, 0.5)

        # assert(np.any(start_mask))

        # end_mask = is_inside_mask(
        #     np.asarray([s[-1] for s in streamlines])[:, None],
        #     self.target_mask.data, affine_vox2mask, 0.5)

        # assert(np.any(end_mask))

        # mask_mask = np.logical_not(np.logical_and(start_mask, end_mask))

        # dirty_mask = np.logical_or(
        #     dirty_mask,
        #     mask_mask)

        # Filter by loops
        # For example: A streamline ending and starting in the same roi
        looping_mask = np.array([winding(s) for s in streamlines]) > 330
        dirty_mask = np.logical_or(
            dirty_mask,
            looping_mask)

        # Only keep valid streamlines
        valid_indices = np.nonzero(np.logical_not(dirty_mask))
        clean_streamlines = streamlines[valid_indices]
        clean_dps = tractogram.data_per_streamline[valid_indices]
        print('Done !')

        print('Kept {}/{} streamlines'.format(len(valid_indices[0]),
                                              len(streamlines)))

        return Tractogram(clean_streamlines,
                          data_per_streamline=clean_dps)

    def run(self):
        """
        Main method where the magic happens
        """
        # Instanciate environment. Actions will be fed to it and new
        # states will be returned. The environment updates the streamline
        # internally

        back_env, env = self.get_test_envs()

        # Get example state to define NN input size
        example_state = env.reset(0, 1)
        self.input_size = example_state.shape[1]

        algs = {'PPO': PPO,
                'TD3': TD3,
                'SAC': SAC,
                'SACAuto': SACAuto,
                'DDPG': DDPG,
                'VPG': VPG}

        rl_alg = algs[self.algorithm]

        # The RL training algorithm
        alg = rl_alg(
            self.input_size,
            3,
            self.hidden_size,
            stochastic=self.stochastic,
            gm_seeding=self.gm_seeding,
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
        self.display(tractogram, env, reward, 0)


def add_test_args(parser):
    parser.add_argument('dataset_file',
                        help='Path to preprocessed datset file (.hdf5)')
    parser.add_argument('subject_id',
                        help='Subject id to fetch from the dataset file')
    parser.add_argument('reference_file',
                        help='Path to binary seeding mask (.nii|.nii.gz)')
    parser.add_argument('ground_truth_folder',
                        help='Path to reference streamlines (.nii|.nii.gz)')
    parser.add_argument('policy',
                        help='Path to the policy')
    parser.add_argument('hyperparameters',
                        help='File containing the hyperparameters for the '
                             'experiment')
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
    experiment = TrackToLearnTest(
        # Dataset params
        args.path,
        args.experiment,
        args.name,
        args.dataset_file,
        args.subject_id,
        args.reference_file,
        args.ground_truth_folder,
        args.valid_noise,
        args.policy,
        args.stochastic,
        args.hyperparameters,
        args.tracking_batch_size,
        args.remove_invalid_streamlines,
        args.step_size,
        args.n_seeds_per_voxel,
        args.min_length,
        args.max_length,
        args.gm_seeding,
        args.fa_map,
    )

    main(experiment)
