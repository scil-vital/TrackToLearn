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
from nibabel.streamlines import detect_format

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
    add_experiment_args,
    add_model_args,
    add_oracle_args,
    add_reward_args,
    add_tracking_args,
    add_tractometer_args)
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

        self.prob = valid_dto['prob']
        self.noise = valid_dto['noise']
        self.agent = valid_dto['agent']
        self.n_actor = valid_dto['n_actor']
        self.npv = valid_dto['npv']
        self.min_length = valid_dto['min_length']
        self.max_length = valid_dto['max_length']

        self.alignment_weighting = valid_dto['alignment_weighting']
        self.straightness_weighting = valid_dto['straightness_weighting']
        self.length_weighting = valid_dto['length_weighting']
        self.target_bonus_factor = valid_dto['target_bonus_factor']
        self.exclude_penalty_factor = valid_dto['exclude_penalty_factor']
        self.angle_penalty_factor = valid_dto['angle_penalty_factor']
        self.coverage_weighting = valid_dto['coverage_weighting']

        # Oracle parameters
        self.oracle_checkpoint = valid_dto['oracle_checkpoint']
        self.dense_oracle_weighting = valid_dto['dense_oracle_weighting']
        self.sparse_oracle_weighting = valid_dto['sparse_oracle_weighting']
        self.oracle_validator = valid_dto['oracle_validator']
        self.oracle_filter = valid_dto['oracle_filter']
        self.oracle_stopping_criterion = \
            valid_dto['oracle_stopping_criterion']

        # Tractometer parameters
        self.tractometer_validator = valid_dto['tractometer_validator']
        self.tractometer_dilate = valid_dto['tractometer_dilate']
        self.tractometer_weighting = valid_dto['tractometer_weighting']

        self.scoring_data = valid_dto['scoring_data']

        self.compute_reward = True

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
            self.epsilon = hyperparams.get('max_angular_error', 90)
            self.hidden_dims = hyperparams['hidden_dims']
            self.n_signal = hyperparams['n_signal']
            self.n_dirs = hyperparams['n_dirs']
            self.interface_seeding = hyperparams['interface_seeding']
            self.cmc = hyperparams.get('cmc', False)
            self.binary_stopping_threshold = hyperparams.get(
                'binary_stopping_threshold', 0.5)
            self.asymmetric = hyperparams.get('asymmetric', False)
            self.no_retrack = hyperparams.get('no_retrack', False)
            self.action_type = hyperparams.get("action_type", "cartesian")
            self.action_size = hyperparams.get("action_size", 3)

        self.comet_experiment = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not valid_dto['cpu']
            else "cpu")

        self.random_seed = valid_dto['rng_seed']
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.rng = np.random.RandomState(seed=self.random_seed)
        random.seed(self.random_seed)

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
        alg.agent.load(self.agent, 'last_model_state')

        tracker = Tracker(
            alg, self.n_actor, self.interface_seeding,
            self.no_retrack, compress=0.0,
            min_length=self.min_length, max_length=self.max_length,
            save_seeds=False)

        out = join(self.experiment_path, "tractogram_{}_{}_{}.tck".format(
            self.experiment, self.id, env.subject_id))

        # Run tracking
        filetype = detect_format(out)
        env.load_subject()
        tractogram = tracker.track(env, filetype)

        reference = get_reference_info(env.reference)

        header = create_tractogram_header(filetype, *reference)

        # Use generator to save the streamlines on-the-fly
        nib.streamlines.save(tractogram, out, header=header)
        # print('Saved {} streamlines'.format(len(tractogram)))


def add_valid_args(parser):
    parser.add_argument('dataset_file',
                        help='Path to preprocessed datset file (.hdf5)')
    parser.add_argument('agent',
                        help='Path to the policy')
    parser.add_argument('hyperparameters',
                        help='File containing the hyperparameters for the '
                             'experiment')
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
    add_reward_args(parser)
    add_valid_args(parser)
    add_tractometer_args(parser)
    add_oracle_args(parser)
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
