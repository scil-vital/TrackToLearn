import functools
import numpy as np
import torch

from dipy.io.stateful_tractogram import StatefulTractogram
from nibabel.streamlines import Tractogram

from TrackToLearn.datasets.utils import (
    convert_length_mm2vox,
)

from TrackToLearn.environments.reward import Reward

from TrackToLearn.environments.stopping_criteria import (
    is_flag_set,
    BinaryStoppingCriterion,
    CmcStoppingCriterion,
    StoppingFlags)

from TrackToLearn.environments.tracking_env import TrackingEnvironment

from TrackToLearn.environments.utils import (
    get_neighborhood_directions,
    is_too_curvy,
    is_too_long)


class BackwardTrackingEnvironment(TrackingEnvironment):
    """ Pre-initialized environment. Tracking will start at the seed from
    flipped half-streamlines.
    """

    def __init__(self, env: TrackingEnvironment, env_dto: dict):

        # Volumes and masks
        self.reference = env.reference
        self.affine_vox2rasmm = env.affine_vox2rasmm
        self.affine_rasmm2vox = env.affine_rasmm2vox

        self.data_volume = env.data_volume
        self.tracking_mask = env.tracking_mask
        self.target_mask = env.target_mask
        self.include_mask = env.include_mask
        self.exclude_mask = env.exclude_mask
        self.peaks = env.peaks

        self.normalize_obs = False  # env_dto['normalize']
        self.obs_rms = None

        self._state_size = None  # to be calculated later

        # Tracking parameters
        self.n_signal = env_dto['n_signal']
        self.n_dirs = env_dto['n_dirs']
        self.theta = theta = env_dto['theta']
        self.cmc = env_dto['cmc']
        self.asymmetric = env_dto['asymmetric']

        step_size_mm = env_dto['step_size']
        min_length_mm = env_dto['min_length']
        max_length_mm = env_dto['max_length']
        add_neighborhood_mm = env_dto['add_neighborhood']

        # Reward parameters
        self.alignment_weighting = env_dto['alignment_weighting']
        self.straightness_weighting = env_dto['straightness_weighting']
        self.length_weighting = env_dto['length_weighting']
        self.target_bonus_factor = env_dto['target_bonus_factor']
        self.exclude_penalty_factor = env_dto['exclude_penalty_factor']
        self.angle_penalty_factor = env_dto['angle_penalty_factor']
        self.compute_reward = env_dto['compute_reward']

        self.rng = env_dto['rng']
        self.device = env_dto['device']

        # Stopping criteria is a dictionary that maps `StoppingFlags`
        # to functions that indicate whether streamlines should stop or not
        self.stopping_criteria = {}
        mask_data = env.tracking_mask.data.astype(np.uint8)

        self.step_size = convert_length_mm2vox(
            step_size_mm,
            self.affine_vox2rasmm)
        self.min_length = min_length_mm
        self.max_length = max_length_mm

        # Compute maximum length
        self.max_nb_steps = int(self.max_length / step_size_mm)
        self.min_nb_steps = int(self.min_length / step_size_mm)

        if self.compute_reward:
            self.reward_function = Reward(
                peaks=self.peaks,
                exclude=self.exclude_mask,
                target=self.target_mask,
                max_nb_steps=self.max_nb_steps,
                theta=self.theta,
                min_nb_steps=self.min_nb_steps,
                asymmetric=self.asymmetric,
                alignment_weighting=self.alignment_weighting,
                straightness_weighting=self.straightness_weighting,
                length_weighting=self.length_weighting,
                target_bonus_factor=self.target_bonus_factor,
                exclude_penalty_factor=self.exclude_penalty_factor,
                angle_penalty_factor=self.angle_penalty_factor,
                scoring_data=None,  # TODO: Add scoring back
                reference=env.reference)

        self.stopping_criteria[StoppingFlags.STOPPING_LENGTH] = \
            functools.partial(is_too_long,
                              max_nb_steps=self.max_nb_steps)

        self.stopping_criteria[
            StoppingFlags.STOPPING_CURVATURE] = \
            functools.partial(is_too_curvy, max_theta=theta)

        if self.cmc:
            cmc_criterion = CmcStoppingCriterion(
                self.include_mask.data,
                self.exclude_mask.data,
                self.affine_vox2rasmm,
                self.step_size,
                self.min_nb_steps)
            self.stopping_criteria[StoppingFlags.STOPPING_MASK] = cmc_criterion
        else:
            binary_criterion = BinaryStoppingCriterion(
                mask_data,
                0.5)
            self.stopping_criteria[StoppingFlags.STOPPING_MASK] = \
                binary_criterion

        # self.stopping_criteria[
        #     StoppingFlags.STOPPING_LOOP] = \
        #     functools.partial(is_looping,
        #                       loop_threshold=300)

        # Convert neighborhood to voxel space
        self.add_neighborhood_vox = None
        if add_neighborhood_mm:
            self.add_neighborhood_vox = convert_length_mm2vox(
                add_neighborhood_mm,
                self.affine_vox2rasmm)
            self.neighborhood_directions = torch.tensor(
                get_neighborhood_directions(
                    radius=self.add_neighborhood_vox),
                dtype=torch.float16).to(self.device)

    @classmethod
    def from_env(
        cls,
        env_dto: dict,
        env: TrackingEnvironment,
    ):
        """ Initialize the environment from a `forward` environment.
        """
        return cls(env, env_dto)

    def reset(self, streamlines: np.ndarray) -> np.ndarray:
        """ Initialize tracking based on half-streamlines.

        Parameters
        ----------
        streamlines : list
            Half-streamlines to initialize environment

        Returns
        -------
        state: numpy.ndarray
            Initial state for RL model
        """

        # Half-streamlines
        self.seeding_streamlines = [s[:] for s in streamlines]
        N = len(streamlines)

        # Jagged arrays ugh
        # This is dirty, clean up asap
        self.half_lengths = np.asarray(
            [len(s) for s in self.seeding_streamlines])
        max_half_len = max(self.half_lengths)
        half_streamlines = np.zeros(
            (N, max_half_len, 3), dtype=np.float32)

        for i, s in enumerate(self.seeding_streamlines):
            le = self.half_lengths[i]
            half_streamlines[i, :le, :] = s

        self.initial_points = np.asarray([s[0] for s in streamlines])

        # Initialize seeds as streamlines
        self.streamlines = np.concatenate((np.zeros(
            (N, self.max_nb_steps + 1, 3),
            dtype=np.float32), half_streamlines), axis=1)

        self.streamlines = np.flip(self.streamlines, axis=1)
        # This means that all streamlines in the batch are limited by the
        # longest half-streamline :(
        self.lengths = np.ones(N, dtype=np.int32) * max_half_len

        # Done flags for tracking backwards
        self.dones = np.full(N, False)
        self.max_half_len = max_half_len
        self.length = max_half_len
        self.continue_idx = np.arange(N)
        self.flags = np.zeros(N, dtype=int)

        # Signal
        return self._format_state(self.streamlines[:, :self.length])

    def get_streamlines(self) -> StatefulTractogram:

        tractogram = Tractogram()
        # Get both parts of the streamlines.
        stopped_streamlines = [self.streamlines[
            i, self.max_half_len - self.half_lengths[i]:self.lengths[i], :]
            for i in range(len(self.streamlines))]

        # Remove last point if the resulting segment had an angle too high.
        flags = is_flag_set(
            self.flags, StoppingFlags.STOPPING_CURVATURE)
        stopped_streamlines = [
            s[:-1] if f else s for f, s in zip(flags, stopped_streamlines)]

        stopped_seeds = self.initial_points

        # Harvested tractogram
        tractogram = Tractogram(
            streamlines=stopped_streamlines,
            data_per_streamline={"seeds": stopped_seeds,
                                 },
            affine_to_rasmm=self.affine_vox2rasmm)

        return tractogram
