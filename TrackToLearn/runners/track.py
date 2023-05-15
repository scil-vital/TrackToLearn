#!/usr/bin/env python
import argparse
import json
import nibabel as nib
import numpy as np
import random
import torch

from argparse import RawTextHelpFormatter

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
    add_tracking_args)
from TrackToLearn.experiment.tracker import Tracker
from TrackToLearn.experiment.ttl import TrackToLearnExperiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert (torch.cuda.is_available())


class TrackToLearnTrack(TrackToLearnExperiment):
    """ TrackToLearn testing script. Should work on any model trained with a
    TrackToLearn experiment
    """

    def __init__(
        self,
        track_dto,
    ):
        """
        """

        self.random_seed = track_dto['rng_seed']
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.rng = np.random.RandomState(seed=self.random_seed)
        random.seed(self.random_seed)

        self.fodf_file = track_dto['fodf']
        self.wm_file = track_dto['wm_mask']

        self.seeding_file = track_dto['seeding_mask']
        self.tracking_file = track_dto['tracking_mask']

        self.dataset_file = None
        self.subject_id = None

        self.reference_file = track_dto['reference_file']
        self.out_tractogram = track_dto['out_tractogram']

        self.valid_noise = track_dto['valid_noise']
        self.policy = track_dto['policy']
        self.n_actor = track_dto['n_actor']
        self.n_seeds_per_voxel = track_dto['n_seeds_per_voxel']
        self.min_length = track_dto['min_length']
        self.max_length = track_dto['max_length']

        self.compress = track_dto['compress']

        self.run_tractometer = False
        self.compute_reward = False
        self.render = False

        self.fa_map = None
        if 'fa_map_file' in track_dto:
            fa_image = nib.load(
                track_dto['fa_map_file'])
            self.fa_map = MRIDataVolume(
                data=fa_image.get_fdata(),
                affine_vox2rasmm=fa_image.affine)

        with open(track_dto['hyperparameters'], 'r') as json_file:
            hyperparams = json.load(json_file)
            self.algorithm = hyperparams['algorithm']
            self.step_size = float(hyperparams['step_size'])
            self.add_neighborhood = hyperparams['add_neighborhood']
            self.voxel_size = float(hyperparams['voxel_size'])
            self.max_angle = hyperparams['max_angle']
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
            self.no_retrack = hyperparams.get('no_retrack', False)

            self.cmc = hyperparams['cmc']
            self.asymmetric = hyperparams['asymmetric']

        self.comet_experiment = None

    def run(self):
        """
        Main method where the magic happens
        """
        # Instanciate environment. Actions will be fed to it and new
        # states will be returned. The environment updates the streamline

        back_env, env = self.get_tracking_envs()

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
        print('Loading {} agent.'.format(self.algorithm))
        # The RL training algorithm
        alg = rl_alg(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            n_actors=self.n_actor,
            rng=self.rng,
            device=device)

        # Load pretrained policies
        alg.policy.load(self.policy, 'last_model_state')

        # Initialize Tracker, which will handle streamline generation
        tracker = Tracker(
            alg, env, back_env, self.n_actor, self.interface_seeding,
            self.no_retrack, compress=self.compress)

        # Run tracking
        tractogram = tracker.track()

        tractogram.affine_to_rasmm = env.affine_vox2rasmm

        filetype = nib.streamlines.detect_format(args.out_tractogram)
        reference = get_reference_info(self.reference_file)
        header = create_tractogram_header(filetype, *reference)

        # Use generator to save the streamlines on-the-fly
        nib.streamlines.save(tractogram, args.out_tractogram, header=header)
        # print('Saved {} streamlines'.format(len(tractogram)))


def add_track_args(parser):
    parser.add_argument('fodf',
                        help='fODF file.')
    parser.add_argument('wm_mask',
                        help='WM mask file')
    parser.add_argument('seeding_mask',
                        help='Seeding mask file')
    parser.add_argument('tracking_mask',
                        help='Tracking mask file')
    parser.add_argument('reference_file',
                        help='Reference anatomy file.')
    parser.add_argument('policy',
                        help='Path to the policy weights.')
    parser.add_argument('hyperparameters',
                        help='File containing the hyperparameters of the '
                             'agent.')
    parser.add_argument('out_tractogram', type=str,
                        help='Output tractogram.')
    parser.add_argument('--n_actor', type=int, default=4096,
                        help='Number of actors to track with')
    parser.add_argument('--fa_map', type=str, default=None,
                        help='FA map to influence STD for probabilistic' +
                             'tracking')
    parser.add_argument('--compress', type=float, default=0,
                        help='Compression factor. If set, should be around'
                        ' 0.1-0.2.')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use gpu or not')
    parser.add_argument('--rng_seed', default=1337, type=int,
                        help='Seed to fix general randomness')


def parse_args():
    """ Generate a tractogram from a trained recurrent model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    add_track_args(parser)
    add_tracking_args(parser, with_step_size=False)

    arguments = parser.parse_args()
    return arguments


def main(experiment):
    """ Main tracking script """
    experiment.run()


if __name__ == '__main__':
    args = parse_args()

    experiment = TrackToLearnTrack(
        vars(args)
    )

    main(experiment)
