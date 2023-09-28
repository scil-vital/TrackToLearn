from argparse import ArgumentParser
from typing import Tuple

from TrackToLearn.algorithms.rl import RLAlgorithm
from TrackToLearn.environments.env import BaseEnv


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

    def get_valid_envs(self) -> Tuple[BaseEnv, BaseEnv]:
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

    def valid(
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


def add_experiment_args(parser: ArgumentParser):
    parser.add_argument('path', type=str,
                        help='Path to experiment')
    parser.add_argument('experiment',
                        help='Name of experiment.')
    parser.add_argument('id', type=str,
                        help='ID of experiment.')
    parser.add_argument('--workspace', type=str, default='TractOracle',
                        help='Comet.ml workspace')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use gpu or not')
    parser.add_argument('--rng_seed', default=1337, type=int,
                        help='Seed to fix general randomness')
    parser.add_argument('--use_comet', action='store_true',
                        help='Use comet to display training or not')
    parser.add_argument('--render', action='store_true',
                        help='Save screenshots of tracking as it goes along.' +
                        'Preferably disabled on non-graphical environments')


def add_data_args(parser: ArgumentParser):
    parser.add_argument('dataset_file',
                        help='Path to preprocessed dataset file (.hdf5)')
    parser.add_argument('subject_id',
                        help='Subject id to fetch from the dataset file')
    parser.add_argument('valid_dataset_file',
                        help='Path to preprocessed dataset file (.hdf5)')
    parser.add_argument('valid_subject_id',
                        help='Subject id to fetch from the dataset file')
    parser.add_argument('reference_file',
                        help='Path to reference anatomy (.nii.gz).')


def add_environment_args(parser: ArgumentParser):
    parser.add_argument('--n_signal', default=1, type=int,
                        help='Signal at the last n positions')
    parser.add_argument('--n_dirs', default=4, type=int,
                        help='Last n steps taken')
    parser.add_argument('--add_neighborhood', default=0.75, type=float,
                        help='Add neighborhood to model input')

    tracking_mask_group = parser.add_mutually_exclusive_group()
    tracking_mask_group.add_argument(
        '--cmc', action='store_true',
        help='If set, use Continuous Mask Criteria to stop tracking.')
    tracking_mask_group.add_argument(
        '--binary_stopping_threshold',
        type=float, default=0.1,
        help='Lower limit for interpolation of tracking mask value.\n'
             'Tracking will stop below this threshold.')
    parser.add_argument('--asymmetric', action='store_true',
                        help='If set, presume asymmetric fODFs when '
                        'computing reward.')
    parser.add_argument('--action_type', default='cartesian', type=str,
                        choices=['cartesian', 'polar', 'discrete'],
                        help='Type of action produced by the agent.')
    parser.add_argument('--sphere', default=None, type=str,
                        choices=[
                            None,
                            'symmetric362',
                            'symmetric642',
                            'symmetric724',
                            'repulsion100',
                            'repulsion200',
                            'repulsion724'],
                        help='Number of possible directions. \'None\' for'
                             '3D directions.')


def add_reward_args(parser: ArgumentParser):
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
    parser.add_argument('--coverage_weighting', default=0.0, type=float,
                        help='Coverage weighting for reward.')


def add_model_args(parser: ArgumentParser):
    parser.add_argument('--n_actor', default=4096, type=int,
                        help='Number of learners')
    parser.add_argument('--hidden_dims', default='1024-1024-1024', type=str,
                        help='Hidden layers of the model')
    parser.add_argument('--load_agent', default=None, type=str,
                        help='Path to pretrained model')


def add_tracking_args(parser: ArgumentParser):
    parser.add_argument('--npv', default=2, type=int,
                        help='Number of random seeds per seeding mask voxel.')
    parser.add_argument('--theta', default=30, type=int,
                        help='Max angle between segments for tracking.')
    parser.add_argument('--epsilon', default=90, type=int,
                        help='Max angle between peaks and segment'
                             ' (angular error).')
    parser.add_argument('--min_length', type=float, default=20.,
                        metavar='m',
                        help='Minimum length of a streamline in mm. '
                        '[%(default)s]')
    parser.add_argument('--max_length', type=float, default=200.,
                        metavar='M',
                        help='Maximum length of a streamline in mm. '
                        '[%(default)s]')
    parser.add_argument('--step_size', default=0.75, type=float,
                        help='Step size for tracking')
    parser.add_argument('--prob', default=0.0, type=float, metavar='sigma',
                        help='Add noise ~ N (0, `prob`) to the agent\'s\n'
                        'output to make tracking more probabilistic.\n'
                        'Should be between 0.0 and 0.1.'
                        '[%(default)s]')
    parser.add_argument('--interface_seeding', action='store_true',
                        help='If set, don\'t track "backwards"')
    parser.add_argument('--no_retrack', action='store_true',
                        help='If set, don\'t retrack backwards')


def add_tractometer_args(parser: ArgumentParser):
    tractom = parser.add_argument_group('Tractometer')
    tractom.add_argument('--scoring_data', type=str,
                         help='Location of the tractometer scoring data.')
    tractom.add_argument('--tractometer_validator', action='store_true',
                         help='Run tractometer during validation to monitor' +
                         ' how the training is doing w.r.t. ground truth.')


def add_oracle_args(parser: ArgumentParser):
    oracle = parser.add_argument_group('Oracle')
    oracle.add_argument('--oracle_checkpoint', type=str,
                        help='Checkpoint file (.ckpt) of the Oracle')
    oracle.add_argument('--oracle_validator', action='store_true',
                        help='Run a TractOracle model during validation to '
                        'monitor how the training is doing.')
    oracle.add_argument('--oracle_stopping_criterion', action='store_true',
                        help='Stop streamlines according to the Oracle.')
    oracle.add_argument('--oracle_filter', action='store_true',
                        help='Filter streamlines according to the Oracle.')
    rew_oracle = oracle.add_mutually_exclusive_group()
    rew_oracle.add_argument('--dense_oracle_weighting', default=0, type=float,
                            help='Dense oracle weighting for reward.')
    rew_oracle.add_argument('--sparse_oracle_weighting', default=0, type=float,
                            help='Sparse oracle weighting for reward.')
