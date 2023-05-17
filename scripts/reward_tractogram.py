import argparse
import numpy as np
import torch

from dipy.io.streamline import load_tractogram
from tqdm import tqdm

from TrackToLearn.environments.reward import Reward
from TrackToLearn.environments.tracker import Tracker


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_reward(tractogram_file, reward_function):

    def reward_streamline(reward_function, streamlines):
        reward = 0
        lens = np.asarray([len(s) for s in streamlines])
        max_lens = max(lens)
        not_dones = np.asarray([True] * len(lens))
        for i in tqdm(range(2, max_lens)):
            streamlines = streamlines[not_dones]
            new_dones = np.asarray([i] * len(lens)) == (np.asarray(lens) - 1)
            # TODO: Verif how reward is calculated
            # TODO: actually retrack streamlines ?
            reward += np.sum(reward_function(np.asarray(
                [s[:i] for s in streamlines]), ~not_dones))
            lens = lens[~new_dones]
            not_dones = ~new_dones

        return reward

    tractogram = load_tractogram(
        tractogram_file, 'same', bbox_valid_check=False,
        trk_header_check=False)
    tractogram.to_vox()
    reward = reward_streamline(
        reward_function, tractogram.streamlines) / len(tractogram.streamlines)

    return np.sum(reward)


def get_reward_function(
    env_dto: dict,
    device
):

    env_dto['device'] = device
    env_dto['compute_reward'] = True
    env_dto['rng'] = np.random.RandomState(env_dto['rng_seed'])

    # Forward environment
    env = Tracker.from_dataset(
        env_dto,
        'training')

    reward_function = Reward(
        peaks=env.peaks,
        exclude=env.exclude_mask,
        target=env.target_mask,
        max_nb_steps=env.max_nb_steps,
        theta=env.theta,
        min_nb_steps=env.min_nb_steps,
        asymmetric=env.asymmetric,
        alignment_weighting=env.alignment_weighting,
        straightness_weighting=env.straightness_weighting,
        length_weighting=env.length_weighting,
        target_bonus_factor=env.target_bonus_factor,
        exclude_penalty_factor=env.exclude_penalty_factor,
        angle_penalty_factor=env.angle_penalty_factor,
        affine_vox2mask=env.affine_vox2mask,
        scoring_data=None,  # TODO: Add scoring back
        reference=env.reference)

    return reward_function


def buildArgsParser():
    p = argparse.ArgumentParser(description="",
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('tractograms', metavar='TRACTS', type=str, nargs='+',
                   help='Tractogram file')
    p.add_argument('dataset_file',
                   help='Path to preprocessed dataset file (.hdf5)')
    p.add_argument('subject_id',
                   help='Subject id to fetch from the dataset file')
    p.add_argument('--n_signal', default=1, type=int,
                   help='Signal at the last n positions')
    p.add_argument('--n_dirs', default=4, type=int,
                   help='Last n steps taken')
    p.add_argument('--add_neighborhood', default=0.75, type=float,
                   help='Add neighborhood to model input')
    p.add_argument('--npv', default=2, type=int,
                   help='Number of random seeds per seeding mask voxel')
    p.add_argument('--theta', default=30, type=int,
                   help='Max angle for tracking')
    p.add_argument('--min_length', default=20, type=int,
                   help='Minimum length for tracts')
    p.add_argument('--max_length', default=200, type=int,
                   help='Maximum length for tracts')
    p.add_argument('--alignment_weighting', default=1, type=float,
                   help='Alignment weighting for reward')
    p.add_argument('--straightness_weighting', default=0, type=float,
                   help='Straightness weighting for reward')
    p.add_argument('--length_weighting', default=0, type=float,
                   help='Length weighting for reward')
    p.add_argument('--target_bonus_factor', default=0, type=float,
                   help='Bonus for streamlines reaching the target mask')
    p.add_argument('--exclude_penalty_factor', default=0, type=float,
                   help='Penalty for streamlines reaching the exclusion '
                   'mask')
    p.add_argument('--angle_penalty_factor', default=0, type=float,
                   help='Penalty for looping or too-curvy streamlines')
    p.add_argument('--step_size', default=0.75, type=float,
                   help='Step size for tracking')
    p.add_argument('--interface_seeding', action='store_true',
                   help='If set, don\'t track "backwards"')
    p.add_argument('--no_retrack', action='store_true',
                   help='If set, don\'t retrack backwards')
    p.add_argument('--cmc', action='store_true',
                   help='If set, use Continuous Mask Criteria to stop'
                   'tracking.')
    p.add_argument('--asymmetric', action='store_true',
                   help='If set, presume asymmetric fODFs when '
                   'computing reward.')
    p.add_argument('--rng_seed', default=1337, type=int,
                   help='Seed to fix general randomness')

    return p


def main():
    p = buildArgsParser()
    args = p.parse_args()
    tractograms = args.tractograms
    reward_function = get_reward_function(
        vars(args), device)

    returns = []
    for tractogram in tractograms:
        print('Computing return for', tractogram)
        returns.append(compute_reward(tractogram, reward_function))
        print(returns[-1])

    print(np.mean(returns), np.std(returns))


if __name__ == '__main__':
    main()
