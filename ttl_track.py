#!/usr/bin/env python
import argparse
import json
import nibabel as nib
import numpy as np
import random
import torch

from argparse import RawTextHelpFormatter
from os.path import join

from dipy.io.utils import get_reference_info, create_tractogram_header
from scilpy.io.utils import (add_sh_basis_args,
                             assert_inputs_exist, assert_outputs_exist,
                             verify_compression_th)
from scilpy.tracking.utils import (add_mandatory_options_tracking,
                                   add_out_options,
                                   verify_streamline_length_options)

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
from TrackToLearn.experiment.tracker import Tracker
from TrackToLearn.experiment.ttl import TrackToLearnExperiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert (torch.cuda.is_available())

DEFAULT_WM_MODEL = 'example_models/SAC_Auto_ISMRM2015_WM/'
DEFAULT_INTERFACE_MODEL = 'example_models/SAC_Auto_ISMRM2015_interface/'


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

        self.in_odf = track_dto['in_odf']
        self.wm_file = track_dto['in_mask']

        self.in_seed = track_dto['in_seed']
        self.in_mask = track_dto['in_mask']

        self.dataset_file = None
        self.subject_id = None

        self.reference_file = track_dto['in_mask']
        self.out_tractogram = track_dto['out_tractogram']

        self.prob = track_dto['prob']
        self.n_actor = track_dto['n_actor']
        self.npv = track_dto['npv']
        self.min_length = track_dto['min_length']
        self.max_length = track_dto['max_length']

        self.compress = track_dto['compress'] or 0.0
        self.sh_basis = track_dto['sh_basis']
        self.save_seeds = track_dto['save_seeds']

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

        self.policy = track_dto['policy']
        self.hyperparameters = track_dto['hyperparameters']

        with open(self.hyperparameters, 'r') as json_file:
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
            self.interface_seeding = track_dto['interface'] or \
                hyperparams['interface_seeding']
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
        print('Loading environment.')
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
        print('Tracking with {} agent.'.format(self.algorithm))
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
            self.no_retrack, compress=self.compress,
            save_seeds=self.save_seeds)

        # Run tracking
        tractogram = tracker.track()

        tractogram.affine_to_rasmm = env.affine_vox2rasmm

        filetype = nib.streamlines.detect_format(args.out_tractogram)
        reference = get_reference_info(self.wm_file)
        header = create_tractogram_header(filetype, *reference)

        # Use generator to save the streamlines on-the-fly
        nib.streamlines.save(tractogram, args.out_tractogram, header=header)
        # print('Saved {} streamlines'.format(len(tractogram)))


def add_track_args(parser):

    add_mandatory_options_tracking(parser)

    basis_group = parser.add_argument_group('Basis options')
    add_sh_basis_args(basis_group)
    add_out_options(parser)

    parser.add_argument('--policy', type=str,
                        help='Path to the folder containing .pth files.\n'
                        'If not set, will default to the example '
                        'models.\n'
                        'Example: example_models/SAC_Auto_ISMRM2015_WM/')
    parser.add_argument(
        '--hyperparameters', type=str,
        help='Path to the .json file containing the '
        'hyperparameters of your tracking agent. \n'
        'If not set, will default to the example models.\n'
        'Example: example_models/SAC_Auto_ISMRM2015_WM/hyperparameters.json')

    seed_group = parser.add_argument_group('Seeding options')
    seed_group.add_argument('--npv', type=int, default=1,
                            help='Number of seeds per voxel.')
    seed_group.add_argument('--interface', action='store_true',
                            help='If set, tracking will be presumed to be '
                            'initialized at the WM/GM interface.\n**Be '
                            'careful to provide the proper seeding '
                                 'mask.**')
    track_g = parser.add_argument_group('Tracking options')
    track_g.add_argument('--min_length', type=float, default=10.,
                         metavar='m',
                         help='Minimum length of a streamline in mm. '
                         '[%(default)s]')
    track_g.add_argument('--max_length', type=float, default=300.,
                         metavar='M',
                         help='Maximum length of a streamline in mm. '
                         '[%(default)s]')
    track_g.add_argument('--prob', type=float, default=0.0, metavar='sigma',
                         help='Add noise ~ N (0, `prob`) to the agent\'s\n'
                         'output to make tracking more probabilistic.\n'
                         'Should be between 0.0 and 0.1.'
                         '[%(default)s]')
    track_g.add_argument('--fa_map', type=str, default=None,
                         help='Scale the added noise (see `--prob`) according'
                         '\nto the provided FA map (.nii.gz). Optional.')

    ml_g = parser.add_argument_group('Machine-Learning options')
    ml_g.add_argument('--n_actor', type=int, default=10000, metavar='N',
                      help='Number of streamlines to track simultaneously.\n'
                      'Can be seen as the "batch size". Limited by the'
                      ' size of your GPU and RAM.\n[%(default)s]')
    ml_g.add_argument('--rng_seed', default=1337, type=int,
                      help='Random number generator seed.')


def verify_policy_option(parser, args):

    if (args.policy is not None and args.hyperparameters is None) or \
       (args.policy is None and args.hyperparameters is not None):
        parser.error('You must specify both --policy and --hyperparameters '
                     'arguments.')

    if args.interface and args.policy is None:
        args.policy = DEFAULT_INTERFACE_MODEL
        args.hyperparameters = join(
            DEFAULT_INTERFACE_MODEL, 'hyperparameters.json')
    elif args.policy is None:
        args.policy = DEFAULT_WM_MODEL
        args.hyperparameters = join(
            DEFAULT_WM_MODEL, 'hyperparameters.json')


def parse_args():
    """ Generate a tractogram from a trained model. """
    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)

    add_track_args(parser)

    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_odf, args.in_seed, args.in_mask])
    assert_outputs_exist(parser, args, args.out_tractogram)
    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk or ' +
                     'tck): {0}'.format(args.out_tractogram))

    verify_streamline_length_options(parser, args)
    verify_compression_th(args.compress)
    verify_policy_option(parser, args)

    return args


def main(experiment):
    """ Main tracking script """
    experiment.run()


if __name__ == '__main__':
    args = parse_args()

    experiment = TrackToLearnTrack(
        vars(args)
    )

    main(experiment)
