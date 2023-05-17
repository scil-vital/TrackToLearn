import nibabel as nib
import numpy as np

from typing import Tuple

from TrackToLearn.environments.backward_tracking_env import (
    BackwardTrackingEnvironment)
from TrackToLearn.environments.retracking_env import RetrackingEnvironment
from TrackToLearn.environments.tracking_env import TrackingEnvironment
from TrackToLearn.environments.utils import interpolate_volume_at_coordinates


class NoisyTrackingEnvironment(TrackingEnvironment):

    def __init__(
        self,
        input_volume,
        tracking_mask,
        target_mask,
        seeding_mask,
        peaks,
        env_dto,
        include_mask=None,
        exclude_mask=None,

    ):
        """
        Parameters
        ----------
        env_dto: dict
            Dict containing all arguments
        """

        super().__init__(
            input_volume,
            tracking_mask,
            target_mask,
            seeding_mask,
            peaks,
            env_dto,
            include_mask,
            exclude_mask)

        self.prob = env_dto['prob']
        self.fa_map = None
        if env_dto['fa_map']:
            self.fa_map = env_dto['fa_map'].data
        self.max_action = 1.

    def step(
        self,
        directions: np.ndarray,
    ) -> Tuple[np.ndarray, list, bool, dict]:
        """
        Apply actions and grow streamlines for one step forward
        Calculate rewards and if the tracking is done, and compute new
        hidden states

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

        if self.fa_map is not None and self.prob > 0.:
            idx = self.streamlines[self.continue_idx,
                                   self.length-1].astype(np.int32)

            # Get peaks at streamline end
            fa = interpolate_volume_at_coordinates(
                self.fa_map, idx, mode='constant', order=0)
            noise = ((1. - fa) * self.prob)
        else:
            noise = np.asarray([self.prob] * len(directions))

        directions = (
            directions + self.rng.normal(np.zeros((3, 1)), noise).T)
        return super().step(directions)


class NoisyRetrackingEnvironment(RetrackingEnvironment):

    def __init__(
        self,
        env,
        env_dto,
    ):
        """
        Parameters
        ----------
        env: BaseEnv
            Forward env
        env_dto: dict
            Dict containing all arguments
        """

        super().__init__(env, env_dto)

        self.prob = env_dto['prob']
        self.fa_map = None
        if env_dto['fa_map']:
            self.fa_map = env_dto['fa_map'].data
        self.max_action = 1.

    def step(
        self,
        directions: np.ndarray,
    ) -> Tuple[np.ndarray, list, bool, dict]:
        """
        Apply actions and grow streamlines for one step forward
        Calculate rewards and if the tracking is done, and compute new
        hidden states

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

        if self.fa_map is not None and self.prob > 0.:
            idx = self.streamlines[self.continue_idx,
                                   self.length-1].astype(np.int32)

            # Get peaks at streamline end
            fa = interpolate_volume_at_coordinates(
                self.fa_map, idx, mode='constant', order=0)
            noise = ((1. - fa) * self.prob)
        else:
            noise = np.asarray([self.prob] * len(directions))

        directions = (
            directions + self.rng.normal(np.zeros((3, 1)), noise).T)
        return super().step(directions)


class BackwardNoisyTrackingEnvironment(BackwardTrackingEnvironment):

    def __init__(
        self,
        env,
        env_dto,
    ):
        """
        Parameters
        ----------
        env: BaseEnv
            Forward env
        env_dto: dict
            Dict containing all arguments
        """

        super().__init__(env, env_dto)

        self.prob = env_dto['prob']
        self.fa_map = None
        if env_dto['fa_map']:
            self.fa_map = env_dto['fa_map'].data
        self.max_action = 1.

    def step(
        self,
        directions: np.ndarray,
    ) -> Tuple[np.ndarray, list, bool, dict]:
        """
        Apply actions and grow streamlines for one step forward
        Calculate rewards and if the tracking is done. While tracking
        has not surpassed half-streamlines, replace the tracking step
        taken with the actual streamline segment

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

        if self.fa_map is not None and self.prob > 0.:
            idx = self.streamlines[:, self.length-1].astype(np.int32)

            # Use affine to map coordinates in mask space
            indices_mask = nib.affines.apply_affine(
                np.linalg.inv(self.affine_vox2mask), idx).astype(np.int32)

            # Get peaks at streamline end
            fa = interpolate_volume_at_coordinates(
                self.fa_map, indices_mask, mode='constant', order=0)
            noise = ((1. - fa) * self.prob)
        else:
            noise = np.asarray([self.prob] * len(directions))

        directions = (
            directions + self.rng.normal(np.zeros((3, 1)), noise).T)
        return super().step(directions)
