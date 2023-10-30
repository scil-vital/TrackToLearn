import numpy as np

from scipy.ndimage import map_coordinates, spline_filter
from typing import Tuple

from TrackToLearn.environments.backward_tracking_env import (
    BackwardTrackingEnvironment)
from TrackToLearn.environments.retracking_env import RetrackingEnvironment
from TrackToLearn.environments.tracking_env import TrackingEnvironment


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

        self.noise = env_dto['noise']
        self.fa_map = None
        if env_dto['fa_map']:
            self.fa_map = spline_filter(env_dto['fa_map'].data, order=3)
        self.max_action = 1.

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, list, bool, dict]:
        """
        Apply actions and grow streamlines for one step forward
        Calculate rewards and if the tracking is done, and compute new
        hidden states

        Parameters
        ----------
        actions: np.ndarray
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

        directions = actions

        if self.fa_map is not None and self.noise > 0.:
            idx = self.streamlines[self.continue_idx,
                                   self.length-1].astype(np.int32)

            # Get FA at streamline end
            fa = map_coordinates(
                self.fa_map, idx.T - 0.5, prefilter=False)
            noise = ((1. - fa) * self.noise)
        else:
            noise = self.rng.normal(0., self.noise, size=directions.shape)
        directions = (
            directions + noise)
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

        self.noise = env_dto['noise']
        self.fa_map = None
        if env_dto['fa_map']:
            self.fa_map = spline_filter(env_dto['fa_map'].data, order=3)
        self.max_action = 1.

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, list, bool, dict]:
        """
        Apply actions and grow streamlines for one step forward
        Calculate rewards and if the tracking is done, and compute new
        hidden states

        Parameters
        ----------
        actions: np.ndarray
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

        directions = actions

        if self.fa_map is not None and self.noise > 0.:
            idx = self.streamlines[self.continue_idx,
                                   self.length-1].astype(np.int32)

            # Get FA at streamline end
            fa = map_coordinates(
                self.fa_map, idx.T - 0.5, prefilter=False)
            noise = ((1. - fa) * self.noise)
        else:
            noise = np.asarray([self.noise] * len(directions))

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

        self.noise = env_dto['noise']
        self.fa_map = None
        if env_dto['fa_map']:
            self.fa_map = spline_filter(env_dto['fa_map'].data, order=3)
        self.max_action = 1.

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, list, bool, dict]:
        """
        Apply actions and grow streamlines for one step forward
        Calculate rewards and if the tracking is done. While tracking
        has not surpassed half-streamlines, replace the tracking step
        taken with the actual streamline segment

        Parameters
        ----------
        actions: np.ndarray
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

        directions = actions

        if self.fa_map is not None and self.noise > 0.:
            idx = self.streamlines[:, self.length-1].astype(np.int32)

            # Get FA at streamline end
            fa = map_coordinates(
                self.fa_map, idx.T - 0.5, prefilter=False)
            noise = ((1. - fa) * self.noise)
        else:
            noise = np.asarray([self.noise] * len(directions))

        directions = (
            directions + self.rng.normal(np.zeros((3, 1)), noise).T)
        return super().step(directions)
