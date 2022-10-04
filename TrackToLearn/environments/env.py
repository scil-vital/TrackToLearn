import functools
import h5py
import numpy as np
import nibabel as nib
import torch

from dwi_ml.data.processing.space.world_to_vox import convert_world_to_vox
from nibabel.streamlines import Tractogram
from typing import Callable, Dict, Tuple

from TrackToLearn.datasets.utils import (
    MRIDataVolume,
    SubjectData,
)
from TrackToLearn.environments.reward import (
    Reward
)
from TrackToLearn.environments.utils import (
    BinaryStoppingCriterion,
    CmcStoppingCriterion,
    format_state,
    get_neighborhood_directions,
    # is_looping,
    # has_reached_gm,
    is_too_curvy,
    is_too_long,
    StoppingFlags)


class BaseEnv(object):
    """
    Abstract tracking environment.
    TODO: Add more explanations
    """

    def __init__(
        self,
        input_volume: MRIDataVolume,
        tracking_mask: MRIDataVolume,
        include_mask: MRIDataVolume,
        exclude_mask: MRIDataVolume,
        target_mask: MRIDataVolume,
        seeding_mask: MRIDataVolume,
        peaks: MRIDataVolume,
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
        target_bonus_factor: float = 0.0,
        exclude_penalty_factor: float = 0.0,
        angle_penalty_factor: float = 0.0,
        add_neighborhood: float = 1.5,
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
            Maximum length for streamlines in mm
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
        device: torch.device
            Device to run training on.
            Should always be GPU
        """
        self.data_volume = torch.tensor(
            input_volume.data, dtype=torch.float32, device=device)
        self.tracking_mask = tracking_mask
        self.target_mask = target_mask
        self.include_mask = include_mask
        self.exclude_mask = exclude_mask
        self.peaks = peaks
        # Tracking parameters
        self.n_signal = n_signal
        self.n_dirs = n_dirs
        self.max_angle = max_angle
        self.cmc = cmc

        # Reward parameters
        self.asymmetric = asymmetric
        self.alignment_weighting = alignment_weighting
        self.straightness_weighting = straightness_weighting
        self.length_weighting = length_weighting
        self.target_bonus_factor = target_bonus_factor
        self.exclude_penalty_factor = exclude_penalty_factor
        self.angle_penalty_factor = angle_penalty_factor
        self.compute_reward = compute_reward
        self.ground_truth_folder = ground_truth_folder
        self.reference_file = reference_file

        self.rng = rng
        self.device = device

        # TODO: Clean affine stuff

        # Stopping criteria is a dictionary that maps `StoppingFlags`
        # to functions that indicate whether streamlines should stop or not
        self.stopping_criteria = {}
        self.input_dv_affine_vox2rasmm = input_volume.affine_vox2rasmm
        mask_data = tracking_mask.data.astype(np.uint8)

        seeding_data = seeding_mask.data.astype(np.uint8)

        # Compute the affine to align dwi voxel coordinates with
        # mask voxel coordinates
        affine_rasmm2maskvox = np.linalg.inv(tracking_mask.affine_vox2rasmm)
        # affine_dwivox2maskvox :
        # dwi voxel space => rasmm space => mask voxel space
        affine_dwivox2maskvox = np.dot(
            affine_rasmm2maskvox,
            self.input_dv_affine_vox2rasmm)

        self.affine_vox2mask = affine_dwivox2maskvox

        # Compute affine to bring seeds into DWI voxel space
        # affine_seedsvox2dwivox :
        # seeds voxel space => rasmm space => dwi voxel space
        affine_seedsvox2rasmm = tracking_mask.affine_vox2rasmm
        affine_rasmm2dwivox = np.linalg.inv(self.input_dv_affine_vox2rasmm)
        self.affine_seedsvox2dwivox = np.dot(
            affine_rasmm2dwivox, affine_seedsvox2rasmm)
        self.affine_vox2rasmm = self.input_dv_affine_vox2rasmm
        self.affine_rasmm2vox = np.linalg.inv(self.affine_vox2rasmm)
        # Tracking seeds
        self.seeds = self._get_tracking_seeds_from_mask(
            seeding_data,
            self.affine_seedsvox2dwivox,
            n_seeds_per_voxel,
            self.rng)

        self.step_size_vox = convert_world_to_vox(
            step_size,
            self.input_dv_affine_vox2rasmm)
        self.min_length = min_length
        self.max_length = max_length

        # Compute maximum length
        self.max_nb_steps = int(self.max_length / step_size)
        self.min_nb_steps = int(self.min_length / step_size)

        self.reward_function = Reward(
            peaks=self.peaks,
            exclude=self.exclude_mask,
            target=self.target_mask,
            max_nb_steps=self.max_nb_steps,
            max_angle=self.max_angle,
            min_nb_steps=self.min_nb_steps,
            asymmetric=self.asymmetric,
            alignment_weighting=self.alignment_weighting,
            straightness_weighting=self.straightness_weighting,
            length_weighting=self.length_weighting,
            target_bonus_factor=self.target_bonus_factor,
            exclude_penalty_factor=self.exclude_penalty_factor,
            angle_penalty_factor=self.angle_penalty_factor,
            affine_vox2mask=self.affine_vox2mask,
            ground_truth_folder=self.ground_truth_folder,
            reference=self.reference_file)

        self.stopping_criteria[StoppingFlags.STOPPING_LENGTH] = \
            functools.partial(is_too_long,
                              max_nb_steps=self.max_nb_steps)

        self.stopping_criteria[
            StoppingFlags.STOPPING_CURVATURE] = \
            functools.partial(is_too_curvy,
                              max_theta=max_angle)
        if self.cmc:
            cmc_criterion = CmcStoppingCriterion(
                self.include_mask.data,
                self.exclude_mask.data,
                self.input_dv_affine_vox2rasmm,
                self.step_size_vox,
                self.min_nb_steps)
            self.stopping_criteria[StoppingFlags.STOPPING_MASK] = cmc_criterion
        else:
            binary_criterion = BinaryStoppingCriterion(
                mask_data,
                affine_dwivox2maskvox,
                0.5)
            self.stopping_criteria[StoppingFlags.STOPPING_MASK] = \
                binary_criterion

        # self.stopping_criteria[
        #     StoppingFlags.STOPPING_LOOP] = \
        #     functools.partial(is_looping,
        #                       loop_threshold=300)

        # Convert neighborhood to voxel space
        self.add_neighborhood_vox = None
        if add_neighborhood:
            self.add_neighborhood_vox = convert_world_to_vox(
                step_size,
                self.input_dv_affine_vox2rasmm)
            self.neighborhood_directions = torch.tensor(
                get_neighborhood_directions(
                    radius=self.add_neighborhood_vox),
                dtype=torch.float16).to(device)

    @classmethod
    def from_dataset(
        cls,
        dataset_file: str,
        subject_id: str,
        interface_seeding: bool = False,
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
        compute_reward: bool = True,
        reference_file: str = None,
        ground_truth_folder: str = None,
        cmc: bool = False,
        asymmetric: bool = False,
        device=None
    ):
        (input_volume, tracking_mask, include_mask, exclude_mask, target_mask,
         seeding_mask, peaks) = \
            BaseEnv._load_dataset(dataset_file, subject_id, interface_seeding)

        return cls(
            input_volume,
            tracking_mask,
            include_mask,
            exclude_mask,
            target_mask,
            seeding_mask,
            peaks,
            n_signal=n_signal,
            n_dirs=n_dirs,
            step_size=step_size,
            max_angle=max_angle,
            min_length=min_length,
            max_length=max_length,
            n_seeds_per_voxel=n_seeds_per_voxel,
            rng=rng,
            alignment_weighting=alignment_weighting,
            straightness_weighting=straightness_weighting,
            length_weighting=length_weighting,
            target_bonus_factor=target_bonus_factor,
            exclude_penalty_factor=exclude_penalty_factor,
            angle_penalty_factor=angle_penalty_factor,
            add_neighborhood=add_neighborhood,
            compute_reward=True,
            reference_file=reference_file,
            ground_truth_folder=ground_truth_folder,
            cmc=cmc,
            asymmetric=asymmetric,
            device=device)

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
                                exclude_file)

        return cls(
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

    @classmethod
    def _load_dataset(cls, dataset_file, split_id, interface_seeding=False):
        """ Load data volumes and masks from the HDF5

        Should everything be put into `self` ? Should everything be returned
        instead ?
        """

        # Load input volume
        with h5py.File(
                dataset_file, 'r'
        ) as hdf_file:
            if split_id not in ['training', 'validation', 'testing']:
                split_set = hdf_file
                subject = split_id
            else:
                split_set = hdf_file[split_id]
                subjects = list(split_set.keys())
                subject = subjects[0]
            tracto_data = SubjectData.from_hdf_subject(
                split_set, subject)
            tracto_data.input_dv.subject_id = subject
        input_volume = tracto_data.input_dv

        # Load peaks for reward
        peaks = tracto_data.peaks

        # Load tracking mask
        tracking_mask = tracto_data.wm

        # Load target and exclude masks
        target_mask = tracto_data.gm

        include_mask = tracto_data.include
        exclude_mask = tracto_data.exclude

        if interface_seeding:
            seeding = tracto_data.interface
        else:
            seeding = tracto_data.wm

        return (input_volume, tracking_mask, include_mask, exclude_mask,
                target_mask, seeding, peaks)

    @classmethod
    def _load_files(
        cls,
        signal_file,
        peaks_file,
        seeding_file,
        tracking_file,
        target_file,
        include_file,
        exclude_file
    ):

        signal = nib.load(signal_file)
        peaks = nib.load(peaks_file)
        seeding = nib.load(seeding_file)
        tracking = nib.load(tracking_file)
        target = nib.load(target_file)
        include = nib.load(include_file)
        exclude = nib.load(exclude_file)

        signal_volume = MRIDataVolume(
            signal.get_fdata(), signal.affine, filename=signal_file)
        peaks_volume = MRIDataVolume(peaks.get_fdata(), peaks.affine,
                                     filename=peaks_file)
        seeding_volume = MRIDataVolume(
            seeding.get_fdata(), seeding.affine, filename=seeding_file)
        tracking_volume = MRIDataVolume(
            tracking.get_fdata(), tracking.affine, filename=tracking_file)
        target_volume = MRIDataVolume(
            target.get_fdata(), target.affine, filename=target_file)
        include_volume = MRIDataVolume(
            include.get_fdata(), include.affine, filename=include_file)
        exclude_volume = MRIDataVolume(
            exclude.get_fdata(), exclude.affine, filename=exclude_file)

        return (signal_volume, tracking_volume, include_volume, exclude_volume,
                target_volume, seeding_volume, peaks_volume)

    def _get_tracking_seeds_from_mask(
        self,
        mask: np.ndarray,
        affine_seedsvox2dwivox: np.ndarray,
        n_seeds_per_voxel: int,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """ Given a binary seeding mask, get seeds in DWI voxel
        space using the provided affine

        Parameters
        ----------
        mask : 3D `numpy.ndarray`
            Binary seeding mask
        affine_seedsvox2dwivox : `numpy.ndarray`
        n_seeds_per_voxel : int
        rng : `numpy.random.RandomState`

        Returns
        -------
        seeds : `numpy.ndarray`
        """
        seeds = []
        indices = np.array(np.where(mask)).T
        for idx in indices:
            seeds_in_seeding_voxel = idx + rng.uniform(
                -0.5,
                0.5,
                size=(n_seeds_per_voxel, 3))
            seeds_in_dwi_voxel = nib.affines.apply_affine(
                affine_seedsvox2dwivox,
                seeds_in_seeding_voxel)
            seeds.extend(seeds_in_dwi_voxel)
        seeds = np.array(seeds, dtype=np.float16)
        return seeds

    def _format_state(
        self,
        streamlines: np.ndarray
    ) -> np.ndarray:
        """
        From the last streamlines coordinates, extract the corresponding
        SH coefficients

        Parameters
        ----------
        streamlines: `numpy.ndarry`
            Streamlines from which to get the coordinates

        Returns
        -------
        signal: `numpy.ndarray`
            SH coefficients at the coordinates
        """

        return format_state(
            streamlines,
            self.data_volume,
            self.add_neighborhood_vox,
            self.neighborhood_directions,
            self.n_signal,
            self.n_dirs,
            self.device)

    def _filter_stopping_streamlines(
        self,
        streamlines: np.ndarray,
        stopping_criteria: Dict[StoppingFlags, Callable]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Checks which streamlines should stop and which ones should continue.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        stopping_criteria : dict of int->Callable
            List of functions that take as input streamlines, and output a
            boolean numpy array indicating which streamlines should stop

        Returns
        -------
        should_continue : `numpy.ndarray`
            Indices of the streamlines that should continue
        should_stop : `numpy.ndarray`
            Indices of the streamlines that should stop
        flags : `numpy.ndarray`
            `StoppingFlags` that triggered stopping for each stopping
            streamline
        """
        idx = np.arange(len(streamlines))

        should_stop = np.zeros(len(idx), dtype=np.bool_)
        flags = np.zeros(len(idx), dtype=np.uint8)

        # For each possible flag, determine which streamline should stop and
        # keep track of the triggered flag
        for flag, stopping_criterion in stopping_criteria.items():
            stopped_by_criterion = stopping_criterion(streamlines)
            should_stop[stopped_by_criterion] = True
            flags[stopped_by_criterion] |= flag.value

        should_continue = np.logical_not(should_stop)

        return idx[should_continue], idx[should_stop], flags[should_stop]

    def _is_stopping():
        """ Check which streamlines should stop or not according to the
        predefined stopping criteria
        """
        pass

    def _get_from_flag(self, flag: StoppingFlags) -> np.ndarray:
        """ Get streamlines that stopped only for a given stopping flag

        Parameters
        ----------
        flag : `StoppingFlags` object

        Returns
        -------
        stopping_idx : `numpy.ndarray`
            The indices corresponding to the streamlines stopped
            by the provided flag
        """
        _, stopping_idx, stopping_flags = self._is_stopping(
            self.streamlines[:, :self.length])
        return stopping_idx[(stopping_flags & flag) != 0]

    def reset():
        """ Initialize tracking seeds and streamlines
        """
        pass

    def step():
        """
        Apply actions and grow streamlines for one step forward
        Calculate rewards and if the tracking is done, and compute new
        hidden states
        """
        pass

    def render(
        self,
        tractogram: Tractogram = None,
        filename: str = None
    ):
        """ Render the streamlines, either directly or through a file
        Might render from "outside" the environment, like for comet

        Parameters:
        -----------
        tractogram: Tractogram, optional
            Object containing the streamlines and seeds
        path: str, optional
            If set, save the image at the specified location instead
            of displaying directly
        """
        from fury import window, actor
        # Might be rendering from outside the environment
        if tractogram is None:
            tractogram = Tractogram(
                streamlines=self.streamlines[:, :self.length],
                data_per_streamline={
                    'seeds': self.starting_points
                })

        # Reshape peaks for displaying
        X, Y, Z, M = self.peaks.data.shape
        peaks = np.reshape(self.peaks.data, (X, Y, Z, 5, M//5))

        # Setup scene and actors
        scene = window.Scene()

        stream_actor = actor.streamtube(tractogram.streamlines)
        peak_actor = actor.peak_slicer(peaks,
                                       np.ones((X, Y, Z, M)),
                                       colors=(0.2, 0.2, 1.),
                                       opacity=0.5)
        dot_actor = actor.dots(tractogram.data_per_streamline['seeds'],
                               color=(1, 1, 1),
                               opacity=1,
                               dot_size=2.5)
        scene.add(stream_actor)
        scene.add(peak_actor)
        scene.add(dot_actor)
        scene.reset_camera_tight(0.95)

        # Save or display scene
        if filename is not None:
            window.snapshot(
                scene,
                fname=filename,
                offscreen=True,
                size=(800, 800))
        else:
            showm = window.ShowManager(scene, reset_camera=True)
            showm.initialize()
            showm.start()
