#!/usr/bin/env python
import argparse
import json
import nibabel as nib
import numpy as np
import random
import torch

from argparse import RawTextHelpFormatter
from os.path import join as pjoin

from dipy.tracking.metrics import length as slength
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.utils import get_reference_info, create_tractogram_header

from TrackToLearn.algorithms.a2c import A2C
from TrackToLearn.algorithms.acktr import ACKTR
from TrackToLearn.algorithms.ddpg import DDPG
from TrackToLearn.algorithms.ppo import PPO
from TrackToLearn.algorithms.trpo import TRPO
from TrackToLearn.algorithms.td3 import TD3
from TrackToLearn.algorithms.sac import SAC
from TrackToLearn.algorithms.sac_auto import SACAuto
from TrackToLearn.algorithms.vpg import VPG
from TrackToLearn.datasets.utils import MRIDataVolume
from TrackToLearn.experiment.experiment import (
    add_environment_args,
    add_experiment_args,
    add_model_args,
    add_tracking_args)
from TrackToLearn.experiment.tracker import Tracker
from TrackToLearn.experiment.ttl import TrackToLearnExperiment


class TrackToLearnValidation(TrackToLearnExperiment):
    """ TrackToLearn validing script. Should work on any model trained with a
    TrackToLearn experiment. This runs tracking on a dataset (hdf5).

    TODO: Make this script as robust as the tracking.
    """

    def __init__(
        self,
        # Dataset params
        valid_dto,
    ):
        """
        """
        self.experiment_path = valid_dto['path']
        self.experiment = valid_dto['experiment']
        self.id = valid_dto['id']
        self.render = False

        self.valid_dataset_file = self.dataset_file = valid_dto['dataset_file']
        self.valid_subject_id = self.subject_id = valid_dto['subject_id']
        self.reference_file = valid_dto['reference_file']
        self.scoring_data = valid_dto['scoring_data']
        self.prob = valid_dto['prob']
        self.policy = valid_dto['policy']
        self.n_actor = valid_dto['n_actor']
        self.npv = valid_dto['npv']
        self.min_length = valid_dto['min_length']
        self.max_length = valid_dto['max_length']
        self.compute_reward = False
        self.run_tractometer = self.scoring_data is not None

        self.fa_map = None
        if valid_dto['fa_map'] is not None:
            fa_image = nib.load(
                valid_dto['fa_map'])
            self.fa_map = MRIDataVolume(
                data=fa_image.get_fdata(),
                affine_vox2rasmm=fa_image.affine)

        with open(valid_dto['hyperparameters'], 'r') as json_file:
            hyperparams = json.load(json_file)
            self.algorithm = hyperparams['algorithm']
            self.step_size = float(hyperparams['step_size'])
            self.add_neighborhood = hyperparams['add_neighborhood']
            self.voxel_size = float(hyperparams['voxel_size'])
            self.theta = hyperparams['max_angle']
            self.alignment_weighting = hyperparams['alignment_weighting']
            self.straightness_weighting = hyperparams['straightness_weighting']
            self.length_weighting = hyperparams['length_weighting']
            self.target_bonus_factor = hyperparams['target_bonus_factor']
            self.exclude_penalty_factor = hyperparams['exclude_penalty_factor']
            self.angle_penalty_factor = hyperparams['angle_penalty_factor']
            self.hidden_dims = hyperparams['hidden_dims']
            self.n_signal = hyperparams['n_signal']
            self.n_dirs = hyperparams['n_dirs']
            self.interface_seeding = hyperparams['interface_seeding']
            self.cmc = hyperparams.get('cmc', False)
            self.asymmetric = hyperparams.get('asymmetric', False)
            self.no_retrack = hyperparams.get('no_retrack', False)

        self.comet_experiment = None
        self.remove_invalid_streamlines = valid_dto[
            'remove_invalid_streamlines']

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not valid_dto['cpu']
            else "cpu")

        self.random_seed = valid_dto['rng_seed']
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
        # looping_mask = np.array([winding(s) for s in streamlines]) > 330
        # dirty_mask = np.logical_or(
        #     dirty_mask,
        #     looping_mask)

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

        back_env, env = self.get_valid_envs()

        # Get example state to define NN input size
        example_state = env.reset(0, 1)
        self.input_size = example_state.shape[1]
        self.action_size = env.get_action_size()

        # Set the voxel size so the agent traverses the same "quantity" of
        # voxels per step as during training.
        tracking_voxel_size = env.get_voxel_size()
        step_size_mm = (tracking_voxel_size / self.voxel_size) * \
            self.step_size

        print("Agent was trained on a voxel size of {}mm and a "
              "step size of {}mm.".format(self.voxel_size, self.step_size))

        print("Subject has a voxel size of {}mm, setting step size to "
              "{}mm.".format(tracking_voxel_size, step_size_mm))

        if back_env:
            back_env.set_step_size(step_size_mm)
        env.set_step_size(step_size_mm)

        # Load agent
        algs = {'VPG': VPG,
                'A2C': A2C,
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
            self.action_size,
            self.hidden_dims,
            n_actors=self.n_actor,
            rng=self.rng,
            device=self.device)

        # Load pretrained policies
        alg.policy.load(self.policy, 'last_model_state')

        # Initialize Tracker, which will handle streamline generation
        tracker = Tracker(
            alg, env, back_env, self.n_actor, self.interface_seeding,
            self.no_retrack, compress=0.0)

        # Run tracking
        tractogram = tracker.track()

        tractogram.affine_to_rasmm = env.affine_vox2rasmm

        filename = pjoin(
            self.experiment_path, "tractogram_{}_{}_{}.trk".format(
                self.experiment, self.id, self.valid_subject_id))

        filetype = nib.streamlines.detect_format(filename)
        reference = get_reference_info(self.reference_file)
        header = create_tractogram_header(filetype, *reference)

        # Use generator to save the streamlines on-the-fly
        nib.streamlines.save(tractogram, filename, header=header)


def add_valid_args(parser):
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
    parser.add_argument('--scoring_data', default=None,
                        help='Path to tractometer files.')
    parser.add_argument('--remove_invalid_streamlines', action='store_true')
    parser.add_argument('--fa_map', type=str, default=None,
                        help='FA map to influence STD for probabilistic' +
                        'tracking')
    parser.add_argument('--valid_theta', type=float, default=None,
                        help='Max valid angle to override the model\'s own.')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU for tracking.')


def parse_args():
    """ Generate a tractogram from a trained recurrent model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    add_experiment_args(parser)
    add_model_args(parser)
    add_valid_args(parser)
    add_environment_args(parser)
    add_tracking_args(parser)

    arguments = parser.parse_args()
    return arguments


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)
    experiment = TrackToLearnValidation(
        # Dataset params
        vars(args),
    )

    experiment.run()


if __name__ == '__main__':
    main()
