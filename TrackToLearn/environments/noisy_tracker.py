import nibabel as nib
import numpy as np

from typing import Tuple

from TrackToLearn.datasets.utils import MRIDataVolume
from TrackToLearn.environments.tracker import BackwardTracker, Tracker
from TrackToLearn.environments.utils import interpolate_volume_at_coordinates


class NoisyTracker(Tracker):

    def __init__(
        self,
        dataset_file: str,
        subject_id: str,
        fa_map: MRIDataVolume = None,
        n_signal: int = 1,
        n_dirs: int = 4,
        step_size: float = 0.2,
        max_angle: float = 45,
        min_length: float = 10,
        max_length: float = 200,
        n_seeds_per_voxel: int = 4,
        rng: np.random.RandomState = None,
        add_neighborhood: float = 1.2,
        valid_noise: float = 0.15,
        compute_reward: bool = True,
        device=None
    ):
        """
        Parameters
        ----------
        input_volume: MRIDataVolume
            Volumetric data containing the SH coefficients
        tracking_mask: MRIDataVolume
            Volumetric mask where tracking is allowed
        seeding_mask: MRIDataVolume
            Mask where seeding should be done
        target_mask: MRIDataVolume
            Mask representing the tracking endpoints
        exclude_mask: MRIDataVolume
            Mask representing the tracking no-go zones
        peaks: MRIDataVolume
            Volume containing the fODFs peaks
        fa_map: MRIDataVolume
            Volume containing the FA map to influence
            probabilistic tracking
        n_signal: int
            Number of signal "history" to keep in input.
            Similar to using last n-frames in vision task
        n_dirs: int
            Number of last actions to append to input
        step_size: float
            Step size for tracking
        max_angle: float
            Maximum angle for tracking
        min_length: int
            Minimum length for streamlines
        max_length: int
            Maximum length for streamlines
        n_seeds_per_voxel: int
            How many seeds to generate per voxel
        rng : `numpy.random.RandomState`
            Random number generator
        add_neighborhood: float
            Use signal in neighboring voxels for model input
        valid_noise: float
            STD of noise to add to directions taken
        device: torch.device
            Device to run training on.
            Should always be GPU
        """
        self.valid_noise = valid_noise
        self.fa_map = None
        if fa_map:
            self.fa_map = fa_map.data
        self.max_action = 1.

        super().__init__(
            dataset_file,
            subject_id,
            n_signal,
            n_dirs,
            step_size,
            max_angle,
            min_length,
            max_length,
            n_seeds_per_voxel,
            rng,
            add_neighborhood,
            compute_reward,
            device
        )

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

        if self.fa_map is not None and self.valid_noise > 0.:
            idx = self.streamlines[:, self.length-1].astype(np.int)

            # Use affine to map coordinates in mask space
            indices_mask = nib.affines.apply_affine(
                np.linalg.inv(self.affine_vox2mask), idx).astype(np.int)

            # Get peaks at streamline end
            fa = interpolate_volume_at_coordinates(
                self.fa_map, indices_mask, mode='constant', order=0)
            noise = ((1. - fa) * self.valid_noise)
        else:
            noise = np.asarray([self.valid_noise] * len(directions))

        directions = (
            directions + self.rng.normal(np.zeros((3, 1)), noise).T)
        return super().step(directions)


class BackwardNoisyTracker(BackwardTracker):

    def __init__(
        self,
        dataset_file: str,
        subject_id: str,
        fa_map: MRIDataVolume = None,
        n_signal: int = 1,
        n_dirs: int = 4,
        step_size: float = 0.2,
        max_angle: float = 45,
        min_length: float = 10,
        max_length: float = 200,
        n_seeds_per_voxel: int = 4,
        rng: np.random.RandomState = None,
        add_neighborhood: float = 1.2,
        valid_noise: float = 0.15,
        compute_reward: bool = True,
        device=None
    ):
        """
        Parameters
        ----------
        input_volume: MRIDataVolume
            Volumetric data containing the SH coefficients
        tracking_mask: MRIDataVolume
            Volumetric mask where tracking is allowed
        seeding_mask: MRIDataVolume
            Mask where seeding should be done
        target_mask: MRIDataVolume
            Mask representing the tracking endpoints
        exclude_mask: MRIDataVolume
            Mask representing the tracking no-go zones
        peaks: MRIDataVolume
            Volume containing the fODFs peaks
        fa_map: MRIDataVolume
            Volume containing the FA map to influence
            probabilistic tracking
        n_signal: int
            Number of signal "history" to keep in input.
            Similar to using last n-frames in vision task
        n_dirs: int
            Number of last actions to append to input
        step_size: float
            Step size for tracking
        max_angle: float
            Maximum angle for tracking
        min_length: int
            Minimum length for streamlines
        max_length: int
            Maximum length for streamlines
        n_seeds_per_voxel: int
            How many seeds to generate per voxel
        rng : `numpy.random.RandomState`
            Random number generator
        add_neighborhood: float
            Use signal in neighboring voxels for model input
        valid_noise: float
            STD of noise to add to directions taken
        device: torch.device
            Device to run training on.
            Should always be GPU
        """
        self.valid_noise = valid_noise
        self.max_action = 1.
        self.fa_map = None
        if fa_map:
            self.fa_map = fa_map.data

        super().__init__(
            dataset_file,
            subject_id,
            n_signal,
            n_dirs,
            step_size,
            max_angle,
            min_length,
            max_length,
            n_seeds_per_voxel,
            rng,
            add_neighborhood,
            compute_reward,
            device
        )

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

        if self.fa_map is not None and self.valid_noise > 0.:
            idx = self.streamlines[:, self.length-1].astype(np.int)

            # Use affine to map coordinates in mask space
            indices_mask = nib.affines.apply_affine(
                np.linalg.inv(self.affine_vox2mask), idx).astype(np.int)

            # Get peaks at streamline end
            fa = interpolate_volume_at_coordinates(
                self.fa_map, indices_mask, mode='constant', order=0)
            noise = ((1. - fa) * self.valid_noise)
        else:
            noise = np.asarray([self.valid_noise] * len(directions))

        directions = (
            directions + self.rng.normal(np.zeros((3, 1)), noise).T)
        return super().step(directions)
