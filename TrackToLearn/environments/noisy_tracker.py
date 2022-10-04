import nibabel as nib
import numpy as np

from typing import Tuple

from TrackToLearn.datasets.utils import MRIDataVolume
from TrackToLearn.environments.tracker import (
    BackwardTracker, Retracker, Tracker)
from TrackToLearn.environments.env import BaseEnv
from TrackToLearn.environments.utils import interpolate_volume_at_coordinates


class NoisyTracker(Tracker):

    def __init__(
        self,
        input_volume: MRIDataVolume,
        tracking_mask: MRIDataVolume,
        include_mask: MRIDataVolume,
        exclude_mask: MRIDataVolume,
        target_mask: MRIDataVolume,
        seeding_mask: MRIDataVolume,
        peaks: MRIDataVolume,
        fa_map: MRIDataVolume = None,
        n_signal: int = 1,
        n_dirs: int = 4,
        step_size: float = 0.2,
        max_angle: float = 45,
        min_length: float = 10,
        max_length: float = 200,
        n_seeds_per_voxel: int = 4,
        rng: np.random.RandomState = None,
        alignment_weighting: float = 1.,
        straightness_weighting: float = 0.0,
        length_weighting: float = 0.0,
        target_bonus_factor: float = 1.0,
        exclude_penalty_factor: float = -1.0,
        angle_penalty_factor: float = -1.0,
        add_neighborhood: float = 1.2,
        valid_noise: float = 0.15,
        compute_reward: bool = True,
        reference_file: str = None,
        ground_truth_folder: str = None,
        cmc: bool = False,
        asymmetric: bool = False,
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
        alignment_weighting: float
            Reward coefficient for alignment with local odfs peaks
        straightness_weighting: float
            Reward coefficient for streamline straightness
        length_weighting: float
            Reward coefficient for streamline length
        target_bonus_factor: `float`
            Bonus for streamlines reaching the target mask
        exclude_penalty_factor: `float`
            Penalty for streamlines reaching the exclusion mask
        angle_penalty_factor: `float`
            Penalty for looping or too-curvy streamlines
        add_neighborhood: float
            Use signal in neighboring voxels for model input
        valid_noise: float
            STD of noise to add to directions taken
        device: torch.device
            Device to run training on.
            Should always be GPU
        """

        super().__init__(
            input_volume,
            tracking_mask,
            include_mask,
            exclude_mask,
            target_mask,
            seeding_mask,
            peaks,
            n_signal,
            n_dirs,
            step_size,
            max_angle,
            min_length,
            max_length,
            n_seeds_per_voxel,
            rng,
            alignment_weighting,
            straightness_weighting,
            length_weighting,
            target_bonus_factor,
            exclude_penalty_factor,
            angle_penalty_factor,
            add_neighborhood,
            compute_reward,
            reference_file,
            ground_truth_folder,
            cmc,
            asymmetric,
            device)

        self.valid_noise = valid_noise
        self.fa_map = None
        if fa_map:
            self.fa_map = fa_map.data
        self.max_action = 1.

    @classmethod
    def from_dataset(
        cls,
        dataset_file: str,
        subject_id: str,
        interface_seeding: bool = False,
        fa_map: MRIDataVolume = None,
        n_signal: int = 1,
        n_dirs: int = 4,
        step_size: float = 0.2,
        max_angle: float = 45,
        min_length: float = 10,
        max_length: float = 200,
        n_seeds_per_voxel: int = 4,
        rng: np.random.RandomState = None,
        alignment_weighting: float = 1.,
        straightness_weighting: float = 0.0,
        length_weighting: float = 0.0,
        target_bonus_factor: float = 1.0,
        exclude_penalty_factor: float = -1.0,
        angle_penalty_factor: float = -1.0,
        add_neighborhood: float = 1.2,
        valid_noise: float = 0.15,
        compute_reward: bool = True,
        reference_file: str = None,
        ground_truth_folder: str = None,
        cmc: bool = False,
        asymmetric: bool = False,
        device=None
    ):

        (input_volume, tracking_mask, include_mask, exclude_mask, target_mask,
         seeding_mask, peaks) = \
            BaseEnv._load_dataset(dataset_file, 'testing', interface_seeding)

        instance = cls(
            input_volume,
            tracking_mask,
            include_mask,
            exclude_mask,
            target_mask,
            seeding_mask,
            peaks,
            fa_map,
            n_signal,
            n_dirs,
            step_size,
            max_angle,
            min_length,
            max_length,
            n_seeds_per_voxel,
            rng,
            alignment_weighting,
            straightness_weighting,
            length_weighting,
            target_bonus_factor,
            exclude_penalty_factor,
            angle_penalty_factor,
            add_neighborhood,
            valid_noise,
            compute_reward,
            reference_file,
            ground_truth_folder,
            cmc,
            asymmetric,
            device)

        return instance

    @classmethod
    def from_files(
        cls,
        signal_file: str,
        peaks_file: str,
        seeding_file: str,
        tracking_file: str,
        target_file: str,
        include_file: str,
        exclude_file: str,
        interface_seeding: bool = False,
        fa_map: MRIDataVolume = None,
        n_signal: int = 1,
        n_dirs: int = 4,
        step_size: float = 0.2,
        max_angle: float = 45,
        min_length: float = 10,
        max_length: float = 200,
        n_seeds_per_voxel: int = 4,
        rng: np.random.RandomState = None,
        alignment_weighting: float = 1.,
        straightness_weighting: float = 0.0,
        length_weighting: float = 0.0,
        target_bonus_factor: float = 1.0,
        exclude_penalty_factor: float = -1.0,
        angle_penalty_factor: float = -1.0,
        add_neighborhood: float = 1.5,
        valid_noise: float = 0.15,
        compute_reward: bool = True,
        reference_file: str = None,
        ground_truth_folder: str = None,
        cmc: bool = False,
        asymmetric: bool = False,
        device=None
    ):

        (input_volume, tracking_mask, include_mask, exclude_mask, target_mask,
         seeding_mask, peaks) = \
            BaseEnv._load_files(signal_file,
                                peaks_file,
                                seeding_file,
                                tracking_file,
                                target_file,
                                include_file,
                                exclude_file,
                                interface_seeding)

        instance = cls(
            input_volume,
            tracking_mask,
            include_mask,
            exclude_mask,
            target_mask,
            seeding_mask,
            peaks,
            fa_map,
            n_signal,
            n_dirs,
            step_size,
            max_angle,
            min_length,
            max_length,
            n_seeds_per_voxel,
            rng,
            alignment_weighting,
            straightness_weighting,
            length_weighting,
            target_bonus_factor,
            exclude_penalty_factor,
            angle_penalty_factor,
            add_neighborhood,
            valid_noise,
            compute_reward,
            reference_file,
            ground_truth_folder,
            cmc,
            asymmetric,
            device)

        return instance

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


class NoisyRetracker(Retracker):

    def __init__(
        self,
        input_volume: MRIDataVolume,
        tracking_mask: MRIDataVolume,
        include_mask: MRIDataVolume,
        exclude_mask: MRIDataVolume,
        target_mask: MRIDataVolume,
        seeding_mask: MRIDataVolume,
        peaks: MRIDataVolume,
        fa_map: MRIDataVolume = None,
        n_signal: int = 1,
        n_dirs: int = 4,
        step_size: float = 0.2,
        max_angle: float = 45,
        min_length: float = 10,
        max_length: float = 200,
        n_seeds_per_voxel: int = 4,
        rng: np.random.RandomState = None,
        alignment_weighting: float = 1.,
        straightness_weighting: float = 0.0,
        length_weighting: float = 0.0,
        target_bonus_factor: float = 1.0,
        exclude_penalty_factor: float = -1.0,
        angle_penalty_factor: float = -1.0,
        add_neighborhood: float = 1.2,
        valid_noise: float = 0.15,
        compute_reward: bool = True,
        reference_file: str = None,
        ground_truth_folder: str = None,
        cmc: bool = False,
        asymmetric: bool = False,
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
        alignment_weighting: float
            Reward coefficient for alignment with local odfs peaks
        straightness_weighting: float
            Reward coefficient for streamline straightness
        length_weighting: float
            Reward coefficient for streamline length
        target_bonus_factor: `float`
            Bonus for streamlines reaching the target mask
        exclude_penalty_factor: `float`
            Penalty for streamlines reaching the exclusion mask
        angle_penalty_factor: `float`
            Penalty for looping or too-curvy streamlines
        add_neighborhood: float
            Use signal in neighboring voxels for model input
        valid_noise: float
            STD of noise to add to directions taken
        device: torch.device
            Device to run training on.
            Should always be GPU
        """

        super().__init__(
            input_volume,
            tracking_mask,
            include_mask,
            exclude_mask,
            target_mask,
            seeding_mask,
            peaks,
            n_signal,
            n_dirs,
            step_size,
            max_angle,
            min_length,
            max_length,
            n_seeds_per_voxel,
            rng,
            alignment_weighting,
            straightness_weighting,
            length_weighting,
            target_bonus_factor,
            exclude_penalty_factor,
            angle_penalty_factor,
            add_neighborhood,
            compute_reward,
            reference_file,
            ground_truth_folder,
            cmc,
            asymmetric,
            device)

        self.valid_noise = valid_noise
        self.fa_map = None
        if fa_map:
            self.fa_map = fa_map.data
        self.max_action = 1.

    @classmethod
    def from_dataset(
        cls,
        dataset_file: str,
        subject_id: str,
        interface_seeding: bool = False,
        fa_map: MRIDataVolume = None,
        n_signal: int = 1,
        n_dirs: int = 4,
        step_size: float = 0.2,
        max_angle: float = 45,
        min_length: float = 10,
        max_length: float = 200,
        n_seeds_per_voxel: int = 4,
        rng: np.random.RandomState = None,
        alignment_weighting: float = 1.,
        straightness_weighting: float = 0.0,
        length_weighting: float = 0.0,
        target_bonus_factor: float = 1.0,
        exclude_penalty_factor: float = -1.0,
        angle_penalty_factor: float = -1.0,
        add_neighborhood: float = 1.2,
        valid_noise: float = 0.15,
        compute_reward: bool = True,
        reference_file: str = None,
        ground_truth_folder: str = None,
        cmc: bool = False,
        asymmetric: bool = False,
        device=None
    ):

        (input_volume, tracking_mask, include_mask, exclude_mask, target_mask,
         seeding_mask, peaks) = \
            BaseEnv._load_dataset(dataset_file, 'testing', interface_seeding)

        instance = cls(
            input_volume,
            tracking_mask,
            include_mask,
            exclude_mask,
            target_mask,
            seeding_mask,
            peaks,
            fa_map,
            n_signal,
            n_dirs,
            step_size,
            max_angle,
            min_length,
            max_length,
            n_seeds_per_voxel,
            rng,
            alignment_weighting,
            straightness_weighting,
            length_weighting,
            target_bonus_factor,
            exclude_penalty_factor,
            angle_penalty_factor,
            add_neighborhood,
            valid_noise,
            compute_reward,
            reference_file,
            ground_truth_folder,
            cmc,
            asymmetric,
            device)

        return instance

    @classmethod
    def from_files(
        cls,
        signal_file: str,
        peaks_file: str,
        seeding_file: str,
        tracking_file: str,
        target_file: str,
        include_file: str,
        exclude_file: str,
        interface_seeding: bool = False,
        fa_map: MRIDataVolume = None,
        n_signal: int = 1,
        n_dirs: int = 4,
        step_size: float = 0.2,
        max_angle: float = 45,
        min_length: float = 10,
        max_length: float = 200,
        n_seeds_per_voxel: int = 4,
        rng: np.random.RandomState = None,
        alignment_weighting: float = 1.,
        straightness_weighting: float = 0.0,
        length_weighting: float = 0.0,
        target_bonus_factor: float = 1.0,
        exclude_penalty_factor: float = -1.0,
        angle_penalty_factor: float = -1.0,
        add_neighborhood: float = 1.5,
        valid_noise: float = 0.15,
        compute_reward: bool = True,
        reference_file: str = None,
        ground_truth_folder: str = None,
        cmc: bool = False,
        asymmetric: bool = False,
        device=None
    ):

        (input_volume, tracking_mask, include_mask, exclude_mask, target_mask,
         seeding_mask, peaks) = \
            BaseEnv._load_files(signal_file,
                                peaks_file,
                                seeding_file,
                                tracking_file,
                                target_file,
                                include_file,
                                exclude_file,
                                interface_seeding)

        instance = cls(
            input_volume,
            tracking_mask,
            include_mask,
            exclude_mask,
            target_mask,
            seeding_mask,
            peaks,
            fa_map,
            n_signal,
            n_dirs,
            step_size,
            max_angle,
            min_length,
            max_length,
            n_seeds_per_voxel,
            rng,
            alignment_weighting,
            straightness_weighting,
            length_weighting,
            target_bonus_factor,
            exclude_penalty_factor,
            angle_penalty_factor,
            add_neighborhood,
            valid_noise,
            compute_reward,
            reference_file,
            ground_truth_folder,
            cmc,
            asymmetric,
            device)

        return instance

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
        input_volume: MRIDataVolume,
        tracking_mask: MRIDataVolume,
        include_mask: MRIDataVolume,
        exclude_mask: MRIDataVolume,
        target_mask: MRIDataVolume,
        seeding_mask: MRIDataVolume,
        peaks: MRIDataVolume,
        fa_map: MRIDataVolume = None,
        n_signal: int = 1,
        n_dirs: int = 4,
        step_size: float = 0.2,
        max_angle: float = 45,
        min_length: float = 10,
        max_length: float = 200,
        n_seeds_per_voxel: int = 4,
        rng: np.random.RandomState = None,
        alignment_weighting: float = 1.,
        straightness_weighting: float = 0.0,
        length_weighting: float = 0.0,
        target_bonus_factor: float = 1.0,
        exclude_penalty_factor: float = -1.0,
        angle_penalty_factor: float = -1.0,
        add_neighborhood: float = 1.2,
        valid_noise: float = 0.15,
        compute_reward: bool = True,
        reference_file: str = None,
        ground_truth_folder: str = None,
        cmc: bool = False,
        asymmetric: bool = False,
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
        alignment_weighting: float
            Reward coefficient for alignment with local odfs peaks
        straightness_weighting: float
            Reward coefficient for streamline straightness
        length_weighting: float
            Reward coefficient for streamline length
        target_bonus_factor: `float`
            Bonus for streamlines reaching the target mask
        exclude_penalty_factor: `float`
            Penalty for streamlines reaching the exclusion mask
        angle_penalty_factor: `float`
            Penalty for looping or too-curvy streamlines
        add_neighborhood: float
            Use signal in neighboring voxels for model input
        valid_noise: float
            STD of noise to add to directions taken
        device: torch.device
            Device to run training on.
            Should always be GPU
        """

        super().__init__(
            input_volume,
            tracking_mask,
            include_mask,
            exclude_mask,
            target_mask,
            seeding_mask,
            peaks,
            n_signal,
            n_dirs,
            step_size,
            max_angle,
            min_length,
            max_length,
            n_seeds_per_voxel,
            rng,
            alignment_weighting,
            straightness_weighting,
            length_weighting,
            target_bonus_factor,
            exclude_penalty_factor,
            angle_penalty_factor,
            add_neighborhood,
            compute_reward,
            reference_file,
            ground_truth_folder,
            cmc,
            asymmetric,
            device)

        self.valid_noise = valid_noise
        self.fa_map = None
        if fa_map:
            self.fa_map = fa_map.data
        self.max_action = 1.

    @classmethod
    def from_dataset(
        cls,
        dataset_file: str,
        subject_id: str,
        interface_seeding: bool = False,
        fa_map: MRIDataVolume = None,
        n_signal: int = 1,
        n_dirs: int = 4,
        step_size: float = 0.2,
        max_angle: float = 45,
        min_length: float = 10,
        max_length: float = 200,
        n_seeds_per_voxel: int = 4,
        rng: np.random.RandomState = None,
        alignment_weighting: float = 1.,
        straightness_weighting: float = 0.0,
        length_weighting: float = 0.0,
        target_bonus_factor: float = 1.0,
        exclude_penalty_factor: float = -1.0,
        angle_penalty_factor: float = -1.0,
        add_neighborhood: float = 1.2,
        valid_noise: float = 0.15,
        compute_reward: bool = True,
        reference_file: str = None,
        ground_truth_folder: str = None,
        cmc: bool = False,
        asymmetric: bool = False,
        device=None
    ):

        (input_volume, tracking_mask, include_mask, exclude_mask, target_mask,
         seeding_mask, peaks) = \
            BaseEnv._load_dataset(dataset_file, 'testing', interface_seeding)

        instance = cls(
            input_volume,
            tracking_mask,
            include_mask,
            exclude_mask,
            target_mask,
            seeding_mask,
            peaks,
            fa_map,
            n_signal,
            n_dirs,
            step_size,
            max_angle,
            min_length,
            max_length,
            n_seeds_per_voxel,
            rng,
            alignment_weighting,
            straightness_weighting,
            length_weighting,
            target_bonus_factor,
            exclude_penalty_factor,
            angle_penalty_factor,
            add_neighborhood,
            valid_noise,
            compute_reward,
            reference_file,
            ground_truth_folder,
            cmc,
            asymmetric,
            device)

        return instance

    @classmethod
    def from_files(
        cls,
        signal_file: str,
        peaks_file: str,
        seeding_file: str,
        tracking_file: str,
        target_file: str,
        include_file: str,
        exclude_file: str,
        interface_seeding: bool = False,
        fa_map: MRIDataVolume = None,
        n_signal: int = 1,
        n_dirs: int = 4,
        step_size: float = 0.2,
        max_angle: float = 45,
        min_length: float = 10,
        max_length: float = 200,
        n_seeds_per_voxel: int = 4,
        rng: np.random.RandomState = None,
        alignment_weighting: float = 1.,
        straightness_weighting: float = 0.0,
        length_weighting: float = 0.0,
        target_bonus_factor: float = 1.0,
        exclude_penalty_factor: float = -1.0,
        angle_penalty_factor: float = -1.0,
        add_neighborhood: float = 1.5,
        valid_noise: float = 0.15,
        compute_reward: bool = True,
        reference_file: str = None,
        ground_truth_folder: str = None,
        cmc: bool = False,
        asymmetric: bool = False,
        device=None
    ):

        (input_volume, tracking_mask, include_mask, exclude_mask, target_mask,
         seeding_mask, peaks) = \
            BaseEnv._load_files(signal_file,
                                peaks_file,
                                seeding_file,
                                tracking_file,
                                target_file,
                                include_file,
                                exclude_file,
                                interface_seeding)

        instance = cls(
            input_volume,
            tracking_mask,
            include_mask,
            exclude_mask,
            target_mask,
            seeding_mask,
            peaks,
            fa_map,
            n_signal,
            n_dirs,
            step_size,
            max_angle,
            min_length,
            max_length,
            n_seeds_per_voxel,
            rng,
            alignment_weighting,
            straightness_weighting,
            length_weighting,
            target_bonus_factor,
            exclude_penalty_factor,
            angle_penalty_factor,
            add_neighborhood,
            valid_noise,
            compute_reward,
            reference_file,
            ground_truth_folder,
            cmc,
            asymmetric,
            device)

        return instance

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
