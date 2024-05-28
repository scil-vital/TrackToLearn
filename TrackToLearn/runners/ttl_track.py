#!/usr/bin/env python3
import argparse
import json
import nibabel as nib
import numpy as np
import os
import random
import torch

from argparse import RawTextHelpFormatter
from os.path import join

from dipy.io.utils import get_reference_info, create_tractogram_header
from nibabel.streamlines import detect_format
from scilpy.io.utils import (add_overwrite_arg,
                             add_sh_basis_args,
                             assert_inputs_exist, assert_outputs_exist,
                             verify_compression_th)
from scilpy.tracking.utils import verify_streamline_length_options

from TrackToLearn.algorithms.sac_auto import SACAuto
from TrackToLearn.datasets.utils import MRIDataVolume

from TrackToLearn.experiment.experiment import Experiment
from TrackToLearn.tracking.tracker import Tracker
from TrackToLearn.utils.torch_utils import get_device

# Define the example model paths from the install folder.
# Hackish ? I'm not aware of a better solution but I'm
# open to suggestions.
_ROOT = os.sep.join(os.path.normpath(
    os.path.dirname(__file__)).split(os.sep)[:-2])
DEFAULT_MODEL = os.path.join(
    _ROOT, 'models')


class TrackToLearnTrack(Experiment):
    """ TrackToLearn testing script. Should work on any model trained with a
    TrackToLearn experiment
    """

    def __init__(
        self,
        track_dto,
    ):
        """
        """

        self.in_odf = track_dto['in_odf']
        self.wm_file = track_dto['in_mask']

        self.in_seed = track_dto['in_seed']
        self.in_mask = track_dto['in_mask']
        self.input_wm = track_dto['input_wm']

        self.dataset_file = None
        self.subject_id = None

        self.reference_file = track_dto['in_mask']
        self.out_tractogram = track_dto['out_tractogram']

        self.noise = track_dto['noise']

        self.binary_stopping_threshold = \
            track_dto['binary_stopping_threshold']

        self.n_actor = track_dto['n_actor']
        self.npv = track_dto['npv']
        self.min_length = track_dto['min_length']
        self.max_length = track_dto['max_length']

        self.compress = track_dto['compress'] or 0.0
        self.sh_basis = track_dto['sh_basis']
        self.save_seeds = track_dto['save_seeds']

        # Tractometer parameters
        self.tractometer_validator = False
        self.scoring_data = None

        self.compute_reward = False
        self.render = False

        self.device = get_device()

        self.fa_map = None
        if 'fa_map_file' in track_dto:
            fa_image = nib.load(
                track_dto['fa_map_file'])
            self.fa_map = MRIDataVolume(
                data=fa_image.get_fdata(),
                affine_vox2rasmm=fa_image.affine)

        self.agent = track_dto['agent']
        self.hyperparameters = track_dto['hyperparameters']

        with open(self.hyperparameters, 'r') as json_file:
            hyperparams = json.load(json_file)
            self.algorithm = hyperparams['algorithm']
            self.step_size = float(hyperparams['step_size'])
            self.voxel_size = hyperparams.get('voxel_size', 2.0)
            self.theta = hyperparams['max_angle']
            self.hidden_dims = hyperparams['hidden_dims']
            self.n_dirs = hyperparams['n_dirs']
            self.target_sh_order = hyperparams['target_sh_order']

        self.alignment_weighting = 0.0
        # Oracle parameters
        self.oracle_checkpoint = None
        self.oracle_bonus = 0.0
        self.oracle_validator = False
        self.oracle_stopping_criterion = False

        self.random_seed = track_dto['rng_seed']
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        self.rng = np.random.RandomState(seed=self.random_seed)

        self.comet_experiment = None

    def run(self):
        """
        Main method where the magic happens
        """
        # Presume iso vox
        ref_img = nib.load(self.reference_file)
        tracking_voxel_size = ref_img.header.get_zooms()[0]

        # # Set the voxel size so the agent traverses the same "quantity" of
        # # voxels per step as during training.
        step_size_mm = self.step_size
        if abs(float(tracking_voxel_size) - float(self.voxel_size)) >= 0.1:
            step_size_mm = (
                float(tracking_voxel_size) / float(self.voxel_size)) * \
                self.step_size

            print("Agent was trained on a voxel size of {}mm and a "
                  "step size of {}mm.".format(self.voxel_size, self.step_size))

            print("Subject has a voxel size of {}mm, setting step size to "
                  "{}mm.".format(tracking_voxel_size, step_size_mm))

        # Instanciate environment. Actions will be fed to it and new
        # states will be returned. The environment updates the streamline
        env = self.get_tracking_env()
        env.step_size_mm = step_size_mm

        # Get example state to define NN input size
        example_state = env.reset(0, 1)
        self.input_size = example_state.shape[1]
        self.action_size = env.get_action_size()

        # Load agent
        algs = {'SACAuto': SACAuto}

        rl_alg = algs[self.algorithm]
        print('Tracking with {} agent.'.format(self.algorithm))
        # The RL training algorithm
        alg = rl_alg(
            self.input_size,
            self.action_size,
            self.hidden_dims,
            n_actors=self.n_actor,
            rng=self.rng,
            device=self.device)

        # Load pretrained policies
        alg.agent.load(self.agent, 'last_model_state')

        # Initialize Tracker, which will handle streamline generation

        tracker = Tracker(
            alg, self.n_actor, compress=self.compress,
            min_length=self.min_length, max_length=self.max_length,
            save_seeds=self.save_seeds)

        # Run tracking
        env.load_subject()
        filetype = detect_format(self.out_tractogram)
        tractogram = tracker.track(env, filetype)

        reference = get_reference_info(self.reference_file)
        header = create_tractogram_header(filetype, *reference)

        # Use generator to save the streamlines on-the-fly
        nib.streamlines.save(tractogram, self.out_tractogram, header=header)


def add_mandatory_options_tracking(p):
    p.add_argument('in_odf',
                   help='File containing the orientation diffusion function \n'
                        'as spherical harmonics file (.nii.gz). Ex: ODF or '
                        'fODF.\nCan be of any order and basis (including "full'
                        '" bases for\nasymmetric ODFs). See also --sh_basis.')
    p.add_argument('in_seed',
                   help='Seeding mask (.nii.gz). Must be represent the WM/GM '
                        'interface.')
    p.add_argument('in_mask',
                   help='Tracking mask (.nii.gz).\n'
                        'Tracking will stop outside this mask.')
    p.add_argument('out_tractogram',
                   help='Tractogram output file (must be .trk or .tck).')
    p.add_argument('--input_wm', action='store_true',
                   help='If set, append the WM mask to the input signal. The '
                        'agent must have been trained accordingly.')


def add_out_options(p):
    out_g = p.add_argument_group('Output options')
    out_g.add_argument('--compress', type=float, metavar='thresh',
                       help='If set, will compress streamlines. The parameter '
                            'value is the \ndistance threshold. A rule of '
                            'thumb is to set it to 0.1mm for \ndeterministic '
                            'streamlines and 0.2mm for probabilitic '
                            'streamlines [%(default)s].')
    add_overwrite_arg(out_g)
    out_g.add_argument('--save_seeds', action='store_true',
                       help='If set, save the seeds used for the tracking \n '
                            'in the data_per_streamline property.\n'
                            'Hint: you can then use '
                            'scilpy_compute_seed_density_map.')
    return out_g


def add_track_args(parser):

    add_mandatory_options_tracking(parser)

    basis_group = parser.add_argument_group('Basis options')
    add_sh_basis_args(basis_group)
    add_out_options(parser)

    agent_group = parser.add_argument_group('Tracking agent options')
    agent_group.add_argument('--agent', type=str,
                             help='Path to the folder containing .pth files.\n'
                             'If not set, will default to the example '
                             'models.\n'
                             '[{}]'.format(DEFAULT_MODEL))
    agent_group.add_argument(
        '--hyperparameters', type=str,
        help='Path to the .json file containing the '
        'hyperparameters of your tracking agent. \n'
        'If not set, will default to the example models.\n'
        '[{}]'.format(DEFAULT_MODEL))
    agent_group.add_argument('--n_actor', type=int, default=10000, metavar='N',
                             help='Number of streamlines to track simultaneous'
                             'ly.\nLimited by the size of your GPU and RAM. A '
                             'higher value\nwill speed up tracking up to a '
                             'point [%(default)s].')

    seed_group = parser.add_argument_group('Seeding options')
    seed_group.add_argument('--npv', type=int, default=1,
                            help='Number of seeds per voxel [%(default)s].')
    track_g = parser.add_argument_group('Tracking options')
    track_g.add_argument('--min_length', type=float, default=10.,
                         metavar='m',
                         help='Minimum length of a streamline in mm. '
                         '[%(default)s]')
    track_g.add_argument('--max_length', type=float, default=300.,
                         metavar='M',
                         help='Maximum length of a streamline in mm. '
                         '[%(default)s]')
    track_g.add_argument('--noise', default=0.0, type=float, metavar='sigma',
                         help='Add noise ~ N (0, `noise`) to the agent\'s\n'
                         'output to make tracking more probabilistic.\n'
                         'Should be between 0.0 and 0.1.'
                         '[%(default)s]')
    track_g.add_argument('--fa_map', type=str, default=None,
                         help='Scale the added noise (see `--noise`) according'
                         '\nto the provided FA map (.nii.gz). Optional.')
    track_g.add_argument(
        '--binary_stopping_threshold',
        type=float, default=0.1,
        help='Lower limit for interpolation of tracking mask value.\n'
             'Tracking will stop below this threshold.')
    parser.add_argument('--rng_seed', default=1337, type=int,
                        help='Random number generator seed [%(default)s].')


def verify_agent_option(parser, args):

    if (args.agent is not None and args.hyperparameters is None) or \
       (args.agent is None and args.hyperparameters is not None):
        parser.error('You must specify both --agent and --hyperparameters '
                     'arguments or use the default model.')

    if args.agent is None:
        args.agent = DEFAULT_MODEL
        args.hyperparameters = join(
            DEFAULT_MODEL, 'hyperparameters.json')


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
    verify_agent_option(parser, args)

    return args


def main():
    """ Main tracking script """
    args = parse_args()

    experiment = TrackToLearnTrack(
        vars(args)
    )

    experiment.run()


if __name__ == '__main__':
    main()
