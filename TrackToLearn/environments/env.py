import functools
import h5py
import os

import numpy as np
import nibabel as nib
import torch

from nibabel.streamlines import Tractogram
from os.path import join as pjoin
from typing import Callable, Dict, Tuple

from TrackToLearn.datasets.utils import (
    convert_length_mm2vox,
    TractographyData,
)
from TrackToLearn.environments.utils import (
    get_neighborhood_directions,
    get_sh,
    is_looping,
    is_too_curvy,
    is_outside_mask,
    is_too_long,
    StoppingFlags)


class BaseEnv(object):
    """
    Abstract tracking environment.
    TODO: Add more explanations
    """

    def __init__(
        self,
        dataset_file: str,
        subject_id: str,
        n_signal: int = 1,
        n_dirs: int = 4,
        step_size: float = 0.2,
        max_angle: float = 45,
        min_length: float = 10,
        max_length: float = 200,
        n_seeds_per_voxel: int = 4,
        rng: np.random.RandomState = None,
        add_neighborhood: float = 1.5,
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
        add_neighborhood: float
            Use signal in neighboring voxels for model input
        device: torch.device
            Device to run training on.
            Should always be GPU
        """

        (input_volume, tracking_mask, exclude_mask, target_mask,
         seeding_mask, peaks) = \
            self._load_dataset(dataset_file, subject_id)

        # Volumes and masks
        self.data_volume = torch.tensor(
            input_volume.data, dtype=torch.float32, device=device)
        self.tracking_mask = tracking_mask
        self.target_mask = target_mask

        self.exclude_mask = exclude_mask
        self.peaks = peaks
        # Tracking parameters
        self.step_size = step_size
        self.n_signal = n_signal
        self.n_dirs = n_dirs
        self.max_angle = max_angle
        self.min_length = int(np.ceil(min_length / step_size))
        self.max_length = int(np.ceil(max_length / step_size))

        self.compute_reward = compute_reward

        self.rng = rng
        self.device = device

        # Stopping criteria is a dictionary that maps `StoppingFlags`
        # to functions that indicate whether streamlines should stop or not
        self.stopping_criteria = {}
        input_dv_affine_vox2rasmm = input_volume.affine_vox2rasmm
        mask_data = tracking_mask.data.astype(np.uint8)

        if seeding_mask is None:
            seeding_data = tracking_mask.data.astype(np.uint8)
        else:
            seeding_data = seeding_mask.data.astype(np.uint8)
        # Compute the affine to align dwi voxel coordinates with
        # mask voxel coordinates
        affine_rasmm2maskvox = np.linalg.inv(tracking_mask.affine_vox2rasmm)
        # affine_dwivox2maskvox :
        # dwi voxel space => rasmm space => mask voxel space
        affine_dwivox2maskvox = np.dot(
            affine_rasmm2maskvox,
            input_dv_affine_vox2rasmm)
        self.affine_vox2mask = affine_dwivox2maskvox

        self.stopping_criteria[StoppingFlags.STOPPING_MASK] = \
            functools.partial(is_outside_mask,
                              mask=mask_data,
                              affine_vox2mask=affine_dwivox2maskvox,
                              threshold=0.5)

        # Compute maximum length
        max_nb_steps = self.max_length
        self.stopping_criteria[StoppingFlags.STOPPING_LENGTH] = \
            functools.partial(is_too_long,
                              max_nb_steps=max_nb_steps)

        self.stopping_criteria[
            StoppingFlags.STOPPING_CURVATURE] = \
            functools.partial(is_too_curvy,
                              max_theta=max_angle)

        self.stopping_criteria[
            StoppingFlags.STOPPING_LOOP] = \
            functools.partial(is_looping,
                              loop_threshold=1750)

        # Convert neighborhood to voxel space
        self.add_neighborhood_vox = None
        if add_neighborhood:
            self.add_neighborhood_vox = convert_length_mm2vox(
                add_neighborhood,
                input_dv_affine_vox2rasmm)
            self.neighborhood_directions = torch.tensor(
                get_neighborhood_directions(
                    radius=self.add_neighborhood_vox),
                dtype=torch.float16).to(device)

        # Compute affine to bring seeds into DWI voxel space
        # affine_seedsvox2dwivox :
        # seeds voxel space => rasmm space => dwi voxel space
        affine_seedsvox2rasmm = tracking_mask.affine_vox2rasmm
        affine_rasmm2dwivox = np.linalg.inv(input_dv_affine_vox2rasmm)
        self.affine_seedsvox2dwivox = np.dot(
            affine_rasmm2dwivox, affine_seedsvox2rasmm)
        self.affine_vox2rasmm = input_dv_affine_vox2rasmm
        self.affine_rasmm2vox = np.linalg.inv(self.affine_vox2rasmm)

        # Tracking seeds
        self.seeds = self._get_tracking_seeds_from_mask(
            seeding_data,
            self.affine_seedsvox2dwivox,
            n_seeds_per_voxel,
            self.rng)

    def _load_dataset(self, dataset_file, subject_id):
        """ Load data volumes and masks from the HDF5

        Should everything be put into `self` ? Should everything be returned
        instead ?
        """

        # Load input volume
        with h5py.File(
                dataset_file, 'r'
        ) as hdf_file:
            assert subject_id in list(
                hdf_file.keys()), (("Subject {} not found in file: {}\n" +
                                    "Subjects are {}").format(
                    subject_id,
                    dataset_file,
                    list(hdf_file.keys())))
            tracto_data = TractographyData.from_hdf_subject(
                hdf_file[subject_id])
            tracto_data.input_dv.subject_id = subject_id
        input_volume = tracto_data.input_dv

        # Load peaks for reward
        peaks = tracto_data.peaks

        # Load tracking mask
        tracking_mask = tracto_data.tracking

        # Load target and exclude masks
        target_mask = tracto_data.target

        exclude_mask = tracto_data.exclude

        seeding = tracto_data.seeding

        return (input_volume, tracking_mask, exclude_mask,
                target_mask, seeding, peaks)

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
        N, L, P = streamlines.shape

        segments = streamlines[:, :-(self.n_signal + 1):-1, :]

        signal = get_sh(
            segments,
            self.data_volume,
            self.add_neighborhood_vox,
            self.neighborhood_directions,
            self.n_signal,
            self.device
        )

        previous_dirs = np.zeros((N, self.n_dirs, P), dtype=np.float32)
        if L > 1:
            dirs = streamlines[:, 1:, :] - streamlines[:, :-1, :]
            previous_dirs[:, :min(dirs.shape[1], self.n_dirs), :] = \
                dirs[:, :-(self.n_dirs+1):-1, :]

        dir_inputs = torch.reshape(torch.as_tensor(previous_dirs,
                                                   device=self.device),
                                   (N, self.n_dirs * P))
        inputs = torch.cat((signal, dir_inputs), dim=-1).to(self.device)
        return inputs.cpu().numpy()

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

    def _keep():
        """ Keep only streamlines corresponding to the given indices, and remove
        all others. The model states will be updated accordingly.
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
            directory = os.path.dirname(pjoin(self.experiment_path, 'render'))
            if not os.path.exists(directory):
                os.makedirs(directory)
            dest = pjoin(directory, filename)
            window.snapshot(
                scene,
                fname=dest,
                offscreen=True,
                size=(800, 800))
        else:
            showm = window.ShowManager(scene, reset_camera=True)
            showm.initialize()
            showm.start()
