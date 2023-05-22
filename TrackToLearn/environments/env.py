import functools
import h5py
import numpy as np
import nibabel as nib
import torch

from gymnasium.wrappers.normalize import RunningMeanStd
from nibabel.streamlines import Tractogram
from typing import Callable, Dict, Tuple

from TrackToLearn.datasets.utils import (
    convert_length_mm2vox,
    MRIDataVolume,
    SubjectData,
    set_sh_order_basis
)

from TrackToLearn.environments.reward import Reward

from TrackToLearn.environments.stopping_criteria import (
    BinaryStoppingCriterion,
    CmcStoppingCriterion,
    StoppingFlags)

from TrackToLearn.environments.utils import (
    get_neighborhood_directions,
    get_sh,
    is_too_curvy,
    is_too_long)


class BaseEnv(object):
    """
    Abstract tracking environment.
    TODO: Add more explanations
    """

    def __init__(
        self,
        input_volume: MRIDataVolume,
        tracking_mask: MRIDataVolume,
        target_mask: MRIDataVolume,
        seeding_mask: MRIDataVolume,
        peaks: MRIDataVolume,
        env_dto: dict,
        include_mask: MRIDataVolume = None,
        exclude_mask: MRIDataVolume = None,
    ):
        """
        Parameters
        ----------
        input_volume: MRIDataVolume
            Volumetric data containing the SH coefficients
        tracking_mask: MRIDataVolume
            Volumetric mask where tracking is allowed
        target_mask: MRIDataVolume
            Mask representing the tracking endpoints
        seeding_mask: MRIDataVolume
            Mask where seeding should be done
        peaks: MRIDataVolume
            Volume containing the fODFs peaks
        env_dto: dict
            DTO containing env. parameters
        include_mask: MRIDataVolume
            Mask representing the tracking go zones. Only useful if
            using CMC.
        exclude_mask: MRIDataVolume
            Mask representing the tracking no-go zones. Only useful if
            using CMC.
        """

        # Volumes and masks
        self.affine_vox2rasmm = input_volume.affine_vox2rasmm
        self.affine_rasmm2vox = np.linalg.inv(self.affine_vox2rasmm)

        self.data_volume = torch.tensor(
            input_volume.data, dtype=torch.float32, device=env_dto['device'])
        self.tracking_mask = tracking_mask
        self.target_mask = target_mask
        self.include_mask = include_mask
        self.exclude_mask = exclude_mask
        self.peaks = peaks

        self.normalize_obs = False  # env_dto['normalize']
        self.obs_rms = None

        self._state_size = None  # to be calculated later

        self.reference = env_dto['reference']

        # Tracking parameters
        self.n_signal = env_dto['n_signal']
        self.n_dirs = env_dto['n_dirs']
        self.theta = theta = env_dto['theta']
        self.npv = env_dto['npv']
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
        mask_data = tracking_mask.data.astype(np.uint8)

        self.seeding_data = seeding_mask.data.astype(np.uint8)

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
                reference=self.reference)

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

        # Tracking seeds
        self.seeds = self._get_tracking_seeds_from_mask(
            self.seeding_data,
            self.npv,
            self.rng)
        print(
            '{} has {} seeds.'.format(self.__class__.__name__,
                                      len(self.seeds)))

    @classmethod
    def from_dataset(
        cls,
        env_dto: dict,
        split: str,
    ):
        dataset_file = env_dto['dataset_file']
        subject_id = env_dto['subject_id']
        interface_seeding = env_dto['interface_seeding']

        (input_volume, tracking_mask, include_mask, exclude_mask, target_mask,
         seeding_mask, peaks) = \
            BaseEnv._load_dataset(
                dataset_file, split, subject_id, interface_seeding
        )

        return cls(
            input_volume,
            tracking_mask,
            target_mask,
            seeding_mask,
            peaks,
            env_dto,
            include_mask,
            exclude_mask,
        )

    @classmethod
    def from_files(
        cls,
        env_dto: dict,
    ):
        in_odf = env_dto['in_odf']
        wm_file = env_dto['wm_file']
        in_seed = env_dto['in_seed']
        in_mask = env_dto['in_mask']
        sh_basis = env_dto['sh_basis']

        input_volume, tracking_mask, seeding_mask = BaseEnv._load_files(
            in_odf,
            wm_file,
            in_seed,
            in_mask,
            sh_basis)

        return cls(
            input_volume,
            tracking_mask,
            None,
            seeding_mask,
            None,
            env_dto)

    @classmethod
    def _load_dataset(
        cls, dataset_file, split_id, subject_id, interface_seeding=False
    ):
        """ Load data volumes and masks from the HDF5

        Should everything be put into `self` ? Should everything be returned
        instead ?
        """

        print("Loading {} from the {} set.".format(subject_id, split_id))
        # Load input volume
        with h5py.File(
                dataset_file, 'r'
        ) as hdf_file:
            print(list(hdf_file.keys()))
            assert split_id in ['training', 'validation', 'testing']
            split_set = hdf_file[split_id]
            tracto_data = SubjectData.from_hdf_subject(
                split_set, subject_id)
            tracto_data.input_dv.subject_id = subject_id
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
            print("Seeding from the interface")
            seeding = tracto_data.interface
        else:
            print("Seeding from the WM.")
            seeding = tracto_data.wm

        return (input_volume, tracking_mask, include_mask, exclude_mask,
                target_mask, seeding, peaks)

    @classmethod
    def _load_files(
        cls,
        signal_file,
        wm_file,
        in_seed,
        in_mask,
        sh_basis
    ):
        signal = nib.load(signal_file)

        # Assert that the subject has iso voxels, else stuff will get
        # complicated
        if not np.allclose(np.mean(signal.header.get_zooms()[:3]),
                           signal.header.get_zooms()[0], atol=1e-03):
            print('WARNING: ODF SH file is not isotropic. Tracking cannot be '
                  'ran robustly. You are entering undefined behavior '
                  'territory.')

        data = set_sh_order_basis(signal.get_fdata(dtype=np.float32),
                                  sh_basis,
                                  target_order=6,
                                  target_basis='descoteaux07')

        seeding = nib.load(in_seed)
        tracking = nib.load(in_mask)
        wm = nib.load(wm_file)
        wm_data = wm.get_fdata()
        if len(wm_data.shape) == 3:
            wm_data = wm_data[..., None]

        signal_data = np.concatenate(
            [data, wm_data], axis=-1)

        signal_volume = MRIDataVolume(
            signal_data, signal.affine, filename=signal_file)

        seeding_volume = MRIDataVolume(
            seeding.get_fdata(), seeding.affine, filename=in_seed)
        tracking_volume = MRIDataVolume(
            tracking.get_fdata(), tracking.affine, filename=in_mask)

        return (signal_volume, tracking_volume, seeding_volume)

    def get_state_size(self):
        example_state = self.reset(0, 1)
        self._state_size = example_state.shape[1]
        return self._state_size

    def get_action_size(self):
        """ TODO: Support spherical actions"""

        return 3

    def get_voxel_size(self):
        """ Returns the voxel size by taking the mean value of the diagonal
        of the affine. This implies that the vox size is always isometric.

        Returns
        -------
        voxel_size: float
            Voxel size in mm.

        """
        diag = np.diagonal(self.affine_vox2rasmm)[:3]
        voxel_size = np.mean(np.abs(diag))

        return voxel_size

    def set_step_size(self, step_size_mm):
        """ Set a different step size (in voxels) than computed by the
        environment. This is necessary when the voxel size between training
        and tracking envs is different.
        """

        self.step_size = convert_length_mm2vox(
            step_size_mm,
            self.affine_vox2rasmm)

        if self.add_neighborhood_vox:
            self.add_neighborhood_vox = convert_length_mm2vox(
                step_size_mm,
                self.affine_vox2rasmm)
            self.neighborhood_directions = torch.tensor(
                get_neighborhood_directions(
                    radius=self.add_neighborhood_vox),
                dtype=torch.float16).to(self.device)

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
                reference=self.reference)

        self.stopping_criteria[StoppingFlags.STOPPING_LENGTH] = \
            functools.partial(is_too_long,
                              max_nb_steps=self.max_nb_steps)

        if self.cmc:
            cmc_criterion = CmcStoppingCriterion(
                self.include_mask.data,
                self.exclude_mask.data,
                self.affine_vox2rasmm,
                self.step_size,
                self.min_nb_steps)
            self.stopping_criteria[StoppingFlags.STOPPING_MASK] = cmc_criterion

    def _normalize(self, obs):
        """Normalises the observation using the running mean and variance of
        the observations. Taken from Gymnasium."""
        if self.obs_rms is None:
            self.obs_rms = RunningMeanStd(shape=(self._state_size,))
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)

    def _get_tracking_seeds_from_mask(
        self,
        mask: np.ndarray,
        npv: int,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """ Given a binary seeding mask, get seeds in DWI voxel
        space using the provided affine. TODO: Replace this
        with scilpy's SeedGenerator

        Parameters
        ----------
        mask : 3D `numpy.ndarray`
            Binary seeding mask
        npv : int
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
                size=(npv, 3))
            seeds.extend(seeds_in_seeding_voxel)
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
        inputs: `numpy.ndarray`
            Observations of the state, incl. previous directions.
        """
        N, L, P = streamlines.shape
        if N <= 0:
            return []
        segments = streamlines[:, -1, :][:, None, :]

        signal = get_sh(
            segments,
            self.data_volume,
            self.add_neighborhood_vox,
            self.neighborhood_directions,
            self.n_signal,
            self.device
        )

        N, S = signal.shape

        inputs = torch.zeros((N, S + (self.n_dirs * P)), device=self.device)

        inputs[:, :S] = signal

        previous_dirs = np.zeros((N, self.n_dirs, P), dtype=np.float32)
        if L > 1:
            dirs = np.diff(streamlines, axis=1)
            previous_dirs[:, :min(dirs.shape[1], self.n_dirs), :] = \
                dirs[:, :-(self.n_dirs+1):-1, :]

        dir_inputs = torch.reshape(
            torch.from_numpy(previous_dirs).to(self.device),
            (N, self.n_dirs * P))

        inputs[:, S:] = dir_inputs

        # if self.normalize_obs and self._state_size is not None:
        #     inputs = self._normalize(inputs)

        return inputs

    def _filter_stopping_streamlines(
        self,
        streamlines: np.ndarray,
        stopping_criteria: Dict[StoppingFlags, Callable]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Checks which streamlines should stop and which ones should
        continue.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        stopping_criteria : dict of int->Callable
            List of functions that take as input streamlines, and output a
            boolean numpy array indicating which streamlines should stop

        Returns
        -------
        should_stop : `numpy.ndarray`
            Boolean array, True is tracking should stop
        flags : `numpy.ndarray`
            `StoppingFlags` that triggered stopping for each stopping
            streamline
        """
        idx = np.arange(len(streamlines))

        should_stop = np.zeros(len(idx), dtype=np.bool_)
        flags = np.zeros(len(idx), dtype=int)

        # For each possible flag, determine which streamline should stop and
        # keep track of the triggered flag
        for flag, stopping_criterion in stopping_criteria.items():
            stopped_by_criterion = stopping_criterion(streamlines)
            flags[stopped_by_criterion] |= flag.value
            should_stop[stopped_by_criterion] = True

        return should_stop, flags

    def _is_stopping():
        """ Check which streamlines should stop or not according to the
        predefined stopping criteria
        """
        pass

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
