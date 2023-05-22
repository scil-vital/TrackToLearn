import functools
import numpy as np
import torch

from typing import Tuple

from TrackToLearn.datasets.utils import (
    convert_length_mm2vox,
)

from TrackToLearn.environments.reward import Reward
from TrackToLearn.environments.tracking_env import TrackingEnvironment

from TrackToLearn.environments.stopping_criteria import (
    BinaryStoppingCriterion,
    CmcStoppingCriterion,
    StoppingFlags)

from TrackToLearn.environments.utils import (
    get_neighborhood_directions,
    is_too_curvy,
    is_too_long)

from TrackToLearn.utils.utils import normalize_vectors


class RetrackingEnvironment(TrackingEnvironment):
    """ Pre-initialized environment
    Tracking will start from the end of streamlines for two reasons:
        - For computational purposes, it's easier if all streamlines have
          the same length and are harvested as they end
        - Tracking back the streamline and computing the alignment allows some
          sort of "self-supervised" learning for tracking backwards
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
        self.scoring_data = env_dto['scoring_data']

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
                scoring_data=self.scoring_data,
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

    def _is_stopping(
        self,
        streamlines: np.ndarray,
        is_still_initializing: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Check which streamlines should stop or not according to the
        predefined stopping criteria. An additional check is performed
        to prevent stopping if the retracking process is not over.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamlines that will be checked
        is_still_initializing: `numpy.ndarray` of shape (n_streamlines)
            Mask that indicates which streamlines are still being
            retracked.

        Returns
        -------
        stopping: `numpy.ndarray`
            Mask of stopping streamlines.
        stopping_flags : `numpy.ndarray`
            `StoppingFlags` that triggered stopping for each stopping
            streamline.
        """
        stopping, flags = super()._is_stopping(streamlines)

        # Streamlines that haven't finished initializing should keep going
        stopping[is_still_initializing[self.continue_idx]] = False
        flags[is_still_initializing[self.continue_idx]] = 0

        return stopping, flags

    def reset(self, half_streamlines: np.ndarray) -> np.ndarray:
        """ Initialize tracking from half-streamlines.

        Parameters
        ----------
        half_streamlines: np.ndarray
            Half-streamlines to initialize environment

        Returns
        -------
        state: numpy.ndarray
            Initial state for RL model
        """

        # Half-streamlines
        self.initial_points = np.array([s[0] for s in half_streamlines])

        # Number if initialization steps for each streamline
        self.n_init_steps = np.asarray(list(map(len, half_streamlines)))

        N = len(self.n_init_steps)

        # Get the first point of each seed as the start of the new streamlines
        self.streamlines = np.zeros(
            (N, self.max_nb_steps, 3),
            dtype=np.float32)

        for i, (s, l) in enumerate(zip(half_streamlines, self.n_init_steps)):
            self.streamlines[i, :l, :] = s[::-1]

        self.seeding_streamlines = self.streamlines.copy()

        self.lengths = np.ones(N, dtype=np.int32)
        self.length = 1

        # Done flags for tracking backwards
        self.flags = np.zeros(N, dtype=int)
        self.dones = np.full(N, False)
        self.continue_idx = np.arange(N)

        # Signal
        return self._format_state(
            self.streamlines[self.continue_idx, :self.length])

    def step(
        self,
        directions: np.ndarray,
    ) -> Tuple[np.ndarray, list, bool, dict]:
        """
        Apply actions and grow streamlines for one step forward
        Calculate rewards and if the tracking is done. While tracking
        has not surpassed half-streamlines, replace the tracking step
        taken with the actual streamline segment.

        Parameters
        ----------
        directions: np.ndarray
            Actions applied to the state

        Returns
        -------
        state: np.ndarray
            New state
        reward: list
            Reward for the last step of the streamline
        done: bool
            Whether the episode is done
        info: dict
        """

        # Scale directions to step size
        directions = normalize_vectors(directions) * self.step_size

        # Grow streamlines one step forward
        self.streamlines[self.continue_idx, self.length,
                         :] = self.streamlines[
                             self.continue_idx, self.length-1, :] + directions
        self.length += 1

        # Check which streamline are still being retracked
        is_still_initializing = self.n_init_steps > self.length + 1

        # Get stopping and keeping indexes
        # self._is_stopping is overridden to take into account retracking
        stopping, new_flags = self._is_stopping(
            self.streamlines[self.continue_idx, :self.length],
            is_still_initializing)

        self.new_continue_idx, self.stopping_idx = (
            self.continue_idx[~stopping],
            self.continue_idx[stopping])

        mask_continue = np.in1d(
            self.continue_idx, self.new_continue_idx, assume_unique=True)
        diff_stopping_idx = np.arange(
            len(self.continue_idx))[~mask_continue]

        # Set "done" flags for RL
        self.dones[self.stopping_idx] = 1

        # Store stopping flags
        self.flags[
            self.stopping_idx] = new_flags[diff_stopping_idx]

        # Compute reward
        reward = np.zeros(self.streamlines.shape[0])
        if self.compute_reward:
            # Reward streamline step
            reward = self.reward_function(
                self.streamlines[self.continue_idx, :self.length, :],
                self.dones[self.continue_idx])

        # If a streamline is still being retracked
        if np.any(is_still_initializing):
            # Replace the last point of the predicted streamlines with
            # the seeding streamlines at the same position

            self.streamlines[is_still_initializing, self.length - 1] = \
                self.seeding_streamlines[is_still_initializing,
                                         self.length - 1]

        # Return relevant infos
        return (
            self._format_state(
                self.streamlines[self.continue_idx, :self.length]),
            reward, self.dones[self.continue_idx],
            {'continue_idx': self.continue_idx})
