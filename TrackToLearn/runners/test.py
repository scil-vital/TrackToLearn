#!/usr/bin/env python
import argparse
import json
import nibabel as nib
import numpy as np
import random
import torch

from argparse import RawTextHelpFormatter
from dipy.tracking.metrics import length as slength, winding
from dipy.io.stateful_tractogram import Space, StatefulTractogram

from TrackToLearn.algorithms.td3 import TD3
from TrackToLearn.algorithms.sac import SAC
from TrackToLearn.algorithms.sac_auto import SACAuto
from TrackToLearn.datasets.utils import MRIDataVolume
from TrackToLearn.runners.experiment import (
    add_environment_args,
    add_experiment_args,
    add_tracking_args)
from TrackToLearn.runners.ttl import TrackToLearnExperiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())


class TrackToLearnTest(TrackToLearnExperiment):
    """ TrackToLearn testing script. Should work on any model trained with a
    TrackToLearn experiment
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
        hyperparameters: str,
        tracking_batch_size: int,
        remove_invalid_streamlines: bool,
        step_size: float,
        n_seeds_per_voxel: int,
        min_length: float,
        max_length: float,
        test_max_angle: float,
        interface_seeding: bool,
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
            if not test_max_angle:
                self.max_angle = hyperparams['max_angle']
            else:
                self.max_angle = test_max_angle
            self.alignment_weighting = 1.0
            self.straightness_weighting = 0.0
            self.length_weighting = 0.0
            self.target_bonus_factor = 0.0
            self.exclude_penalty_factor = 0.0
            self.angle_penalty_factor = 0.0
            self.hidden_dims = hyperparams['hidden_dims']
            self.n_signal = 1
            self.cmc = hyperparams.get('cmc', False)
            self.asymmetric = hyperparams.get('asymmetric', False)
            self.recurrent = hyperparams.get('recurrent', 0)
            self.n_dirs = hyperparams['n_dirs']
            self.interface_seeding = hyperparams['interface_seeding']
            self.no_retrack = hyperparams.get('no_retrack', False)

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
        tractogram = tractogram.to_world()

        streamlines = tractogram.streamlines
        lengths = [slength(s) for s in streamlines]
        # # Filter by curvature
        # dirty_mask = is_flag_set(
        #     stopping_flags, StoppingFlags.STOPPING_CURVATURE)
        dirty_mask = np.zeros(len(streamlines))

        # Filter by length unless the streamline ends in GM
        # Example case: Bundle 3 of fibercup tends to be shorter than 35

        short_lengths = np.asarray(
            [lgt <= self.min_length for lgt in lengths])

        dirty_mask = np.logical_or(short_lengths, dirty_mask)

        long_lengths = np.asarray(
            [lgt > self.max_length for lgt in lengths])

        dirty_mask = np.logical_or(long_lengths, dirty_mask)

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
        sft = StatefulTractogram(
            clean_streamlines,
            self.reference_file,
            space=Space.RASMM,
            data_per_streamline=clean_dps)
        # Rest of the code presumes vox space
        sft.to_vox()
        return sft

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

        algs = {'TD3': TD3,
                'SAC': SAC,
                'SACAuto': SACAuto}

        rl_alg = algs[self.algorithm]

        # The RL training algorithm
        alg = rl_alg(
            self.input_size,
            3,
            self.hidden_dims,
            self.recurrent,
            interface_seeding=self.interface_seeding,
            rng=self.rng,
            device=device)

        # Load pretrained policies
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
    parser.add_argument('policy',
                        help='Path to the policy')
    parser.add_argument('hyperparameters',
                        help='File containing the hyperparameters for the '
                             'experiment')
    parser.add_argument('--remove_invalid_streamlines', action='store_true')
    parser.add_argument('--fa_map', type=str, default=None,
                        help='FA map to influence STD for probabilistic' +
                        'tracking')
    parser.add_argument('--test_max_angle', type=float, default=None,
                        help='Max test angle to override the model\'s own.')


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
        args.hyperparameters,
        args.tracking_batch_size,
        args.remove_invalid_streamlines,
        args.step_size,
        args.n_seeds_per_voxel,
        args.min_length,
        args.max_length,
        args.test_max_angle,
        args.interface_seeding,
        args.fa_map,
    )

    main(experiment)
