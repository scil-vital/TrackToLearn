import functools
from typing import Callable, Dict, Tuple

import nibabel as nib
import numpy as np
import torch
from dipy.core.sphere import HemiSphere
from dipy.data import get_sphere
from dipy.direction.peaks import reshape_peaks_for_visualization
from dipy.tracking import utils as track_utils
from dwi_ml.data.processing.volume.interpolation import \
    interpolate_volume_in_neighborhood
from dwi_ml.data.processing.space.neighborhood import \
    get_neighborhood_vectors_axes
from scilpy.reconst.utils import (find_order_from_nb_coeff, get_b_matrix,
                                  get_maximas)
from torch.utils.data import DataLoader

from TrackToLearn.datasets.SubjectDataset import SubjectDataset
from TrackToLearn.datasets.utils import (MRIDataVolume,
                                         convert_length_mm2vox,
                                         set_sh_order_basis)
from TrackToLearn.environments.connectivity_reward import ConnectivityReward
from TrackToLearn.environments.local_reward import PeaksAlignmentReward
from TrackToLearn.environments.oracle_reward import OracleReward
from TrackToLearn.environments.reward import RewardFunction
from TrackToLearn.environments.stopping_criteria import (
    BinaryStoppingCriterion, OracleStoppingCriterion,
    StoppingFlags)
from TrackToLearn.environments.utils import (  # is_looping,
    is_too_curvy, is_too_long)
from TrackToLearn.utils.utils import normalize_vectors

# from dipy.io.utils import get_reference_info


class BaseEnv(object):
    """
    Abstract tracking environment. This class should not be used directly.
    Instead, use `TrackingEnvironment` or `InferenceTrackingEnvironment`.

    Track-to-Learn environments are based on OpenAI Gym environments. They
    are used to train reinforcement learning algorithms. They also emulate
    "Trackers" in dipy by handling streamline propagation, stopping criteria,
    and seeds.

    Since many streamlines are propagated in parallel, the environment is
    similar to VectorizedEnvironments in the Gym definition. However, the
    environment is not vectorized in the sense that it does not reset
    trajectories (streamlines) independently.

    TODO: reset trajectories independently ?

    """

    def __init__(
        self,
        subject_data: str,
        split_id: str,
        env_dto: dict,
    ):
        """
        Initialize the environment. This should not be called directly.
        Instead, use `from_dataset` or `from_files`.

        Parameters
        ----------
        dataset_file: str
            Path to the HDF5 file containing the dataset.
        split_id: str
            Name of the split to load (e.g. 'training',
            'validation', 'testing').
        subjects: list
            List of subjects to load.
        env_dto: dict
            DTO containing env. parameters

        """

        # If the subject data is a string, it is assumed to be a path to
        # an HDF5 file. Otherwise, it is assumed to be a list of volumes
        if type(subject_data) is str:
            self.dataset_file = subject_data
            self.split = split_id

            def collate_fn(data):
                return data

            self.dataset = SubjectDataset(
                self.dataset_file, self.split)
            self.loader = DataLoader(self.dataset, 1, shuffle=True,
                                     collate_fn=collate_fn,
                                     num_workers=2)
            self.loader_iter = iter(self.loader)
        else:
            self.subject_data = subject_data
            self.split = split_id

        # Unused: this is from an attempt to normalize the input data
        # as is done by the original PPO impl
        # Does not seem to be necessary here.
        self.normalize_obs = False  # env_dto['normalize']
        self.obs_rms = None

        self._state_size = None  # to be calculated later

        # Tracking parameters
        self.n_dirs = env_dto['n_dirs']
        self.theta = env_dto['theta']
        # Number of seeds per voxel
        self.npv = env_dto['npv']
        # Whether to use CMC or binary stopping criterion
        self.binary_stopping_threshold = env_dto['binary_stopping_threshold']

        # Step-size and min/max lengths are typically defined in mm
        # by the user, but need to be converted to voxels.
        self.step_size_mm = env_dto['step_size']
        self.min_length_mm = env_dto['min_length']
        self.max_length_mm = env_dto['max_length']

        # Oracle parameters
        self.oracle_checkpoint = env_dto['oracle_checkpoint']
        self.oracle_stopping_criterion = env_dto['oracle_stopping_criterion']

        # Tractometer parameters
        self.scoring_data = env_dto['scoring_data']

        # Reward parameters
        self.compute_reward = env_dto['compute_reward']
        # "Local" reward parameters
        self.alignment_weighting = env_dto['alignment_weighting']
        # "Sparse" reward parameters
        self.oracle_bonus = env_dto['oracle_bonus']

        # Other parameters
        self.rng = env_dto['rng']
        self.device = env_dto['device']

        # Load one subject as an example
        self.load_subject()

    def load_subject(
        self,
    ):
        """ Load a random subject from the dataset. This is used to
        initialize the environment. """

        if hasattr(self, 'dataset_file'):

            if hasattr(self, 'subject_id') and len(self.dataset) == 1:
                return

            try:
                (sub_id, input_volume, tracking_mask, seeding_mask,
                 peaks, reference, labels, connectivity) = \
                    next(self.loader_iter)[0]
            except StopIteration:
                self.loader_iter = iter(self.loader)
                (sub_id, input_volume, tracking_mask, seeding_mask,
                 peaks, reference, labels, connectivity) = \
                    next(self.loader_iter)[0]

            self.subject_id = sub_id
            # Affines
            self.reference = reference
            self.affine_vox2rasmm = input_volume.affine_vox2rasmm
            self.affine_rasmm2vox = np.linalg.inv(self.affine_vox2rasmm)

            # Volumes and masks
            self.data_volume = torch.from_numpy(
                input_volume.data).to(self.device, dtype=torch.float32)
        else:
            (input_volume, tracking_mask, seeding_mask, peaks,
             reference) = self.subject_data

            self.affine_vox2rasmm = input_volume.affine_vox2rasmm
            self.affine_rasmm2vox = np.linalg.inv(self.affine_vox2rasmm)

            # Volumes and masks
            self.data_volume = torch.from_numpy(
                input_volume.data).to(self.device, dtype=torch.float32)

            self.reference = reference

        self.tracking_mask = tracking_mask
        self.peaks = peaks
        mask_data = tracking_mask.data.astype(np.uint8)
        self.seeding_data = seeding_mask.data.astype(np.uint8)

        self.labels = labels
        self.connectivity = connectivity

        self.step_size = convert_length_mm2vox(
            self.step_size_mm,
            self.affine_vox2rasmm)
        self.min_length = self.min_length_mm
        self.max_length = self.max_length_mm

        # Compute maximum length
        self.max_nb_steps = int(self.max_length / self.step_size_mm)
        self.min_nb_steps = int(self.min_length / self.step_size_mm)

        # Neighborhood used as part of the state
        self.add_neighborhood_vox = convert_length_mm2vox(
            self.step_size_mm,
            self.affine_vox2rasmm)
        self.neighborhood_directions = torch.cat(
            (torch.zeros((1, 3)),
             get_neighborhood_vectors_axes(1, self.add_neighborhood_vox))
        ).to(self.device)

        # Tracking seeds
        self.seeds = track_utils.random_seeds_from_mask(
            self.seeding_data,
            np.eye(4),
            seeds_count=self.npv)
        # print(
        #     '{} has {} seeds.'.format(self.__class__.__name__,
        #                               len(self.seeds)))

        # ===========================================
        # Stopping criteria
        # ===========================================

        # Stopping criteria is a dictionary that maps `StoppingFlags`
        # to functions that indicate whether streamlines should stop or not

        # TODO: Make all stopping criteria classes.
        # TODO?: Use dipy's stopping criteria instead of custom ones ?
        self.stopping_criteria = {}

        # Length criterion
        self.stopping_criteria[StoppingFlags.STOPPING_LENGTH] = \
            functools.partial(is_too_long,
                              max_nb_steps=self.max_nb_steps)

        # Angle between segment (curvature criterion)
        self.stopping_criteria[
            StoppingFlags.STOPPING_CURVATURE] = \
            functools.partial(is_too_curvy, max_theta=self.theta)

        # Stopping criterion according to an oracle
        if self.oracle_checkpoint and self.oracle_stopping_criterion:
            self.stopping_criteria[
                StoppingFlags.STOPPING_ORACLE] = OracleStoppingCriterion(
                self.oracle_checkpoint,
                self.min_nb_steps * 5,
                self.reference,
                self.affine_vox2rasmm,
                self.device)

        # Mask criterion (either binary or CMC)
        binary_criterion = BinaryStoppingCriterion(
            mask_data,
            self.binary_stopping_threshold)
        self.stopping_criteria[StoppingFlags.STOPPING_MASK] = \
            binary_criterion

        # ==========================================
        # Reward function
        # =========================================

        # Reward function and reward factors
        if self.compute_reward:
            # Reward streamline according to alignment with local peaks
            peaks_reward = PeaksAlignmentReward(self.peaks)
            oracle_reward = OracleReward(self.oracle_checkpoint,
                                         self.min_nb_steps,
                                         self.reference,
                                         self.affine_vox2rasmm,
                                         self.device)

            connectivity_reward = ConnectivityReward(self.labels.data,
                                                     self.connectivity,
                                                     self.reference,
                                                     self.affine_vox2rasmm,
                                                     self.min_nb_steps)

            # Combine all reward factors into the reward function
            self.reward_function = RewardFunction(
                [peaks_reward,
                 oracle_reward,
                 connectivity_reward],
                [self.alignment_weighting,
                 self.oracle_bonus,
                 10])

    @classmethod
    def from_dataset(
        cls,
        env_dto: dict,
        split: str,
    ):
        """ Initialize the environment from an HDF5.

        Parameters
        ----------
        env_dto: dict
            DTO containing env. parameters
        split: str
            Name of the split to load (e.g. 'training', 'validation',
            'testing').

        Returns
        -------
        env: BaseEnv
            Environment initialized from a dataset.
        """

        dataset_file = env_dto['dataset_file']

        env = cls(dataset_file, split, env_dto)
        return env

    @classmethod
    def from_files(
        cls,
        env_dto: dict,
    ):
        """ Initialize the environment from files. This is useful for
        tracking from a trained model.

        Parameters
        ----------
        env_dto: dict
            DTO containing env. parameters

        Returns
        -------
        env: BaseEnv
            Environment initialized from files.
        """

        in_odf = env_dto['in_odf']
        in_seed = env_dto['in_seed']
        in_mask = env_dto['in_mask']
        sh_basis = env_dto['sh_basis']
        reference = env_dto['reference']

        (input_volume, peaks_volume, tracking_mask, seeding_mask) = \
            BaseEnv._load_files(
                in_odf,
                in_seed,
                in_mask,
                sh_basis)

        subj_files = (input_volume, tracking_mask, seeding_mask,
                      peaks_volume, reference)

        return cls(subj_files, 'testing', env_dto)

    @classmethod
    def _load_files(
        cls,
        signal_file,
        in_seed,
        in_mask,
        sh_basis,
    ):
        """ Load data volumes and masks from files. This is useful for
        tracking from a trained model.

        If the signal is not in descoteaux07 basis, it will be converted. The
        WM mask will be loaded and concatenated to the signal. Additionally,
        peaks will be computed from the signal.

        Parameters
        ----------
        signal_file: str
            Path to the signal file (e.g. SH coefficients).
        in_seed: str
            Path to the seeding mask.
        in_mask: str
            Path to the tracking mask.
        sh_basis: str
            Basis of the SH coefficients.

        Returns
        -------
        signal_volume: MRIDataVolume
            Volumetric data containing the SH coefficients
        peaks_volume: MRIDataVolume
            Volume containing the fODFs peaks
        tracking_volume: MRIDataVolume
            Volumetric mask where tracking is allowed
        seeding_volume: MRIDataVolume
            Mask where seeding should be done
        """

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
                                  target_order=8,
                                  target_basis='descoteaux07')

        # Compute peaks from signal
        # Does not work if signal is not fODFs
        npeaks = 5
        odf_shape_3d = data.shape[:-1]
        peak_dirs = np.zeros((odf_shape_3d + (npeaks, 3)))
        peak_values = np.zeros((odf_shape_3d + (npeaks, )))

        sphere = HemiSphere.from_sphere(get_sphere("repulsion724")
                                        ).subdivide(0)

        b_matrix = get_b_matrix(
            find_order_from_nb_coeff(data), sphere, "descoteaux07")

        for idx in np.argwhere(np.sum(data, axis=-1)):
            idx = tuple(idx)
            directions, values, indices = get_maximas(data[idx],
                                                      sphere, b_matrix,
                                                      0.1, 0)
            if values.shape[0] != 0:
                n = min(npeaks, values.shape[0])
                peak_dirs[idx][:n] = directions[:n]
                peak_values[idx][:n] = values[:n]

        X, Y, Z, N, P = peak_dirs.shape
        peak_values = np.divide(peak_values, peak_values[..., 0, None],
                                out=np.zeros_like(peak_values),
                                where=peak_values[..., 0, None] != 0)
        peak_dirs[...] *= peak_values[..., :, None]
        peak_dirs = reshape_peaks_for_visualization(peak_dirs)

        # Load rest of volumes
        seeding = nib.load(in_seed)
        tracking = nib.load(in_mask)
        signal_data = data
        signal_volume = MRIDataVolume(
            signal_data, signal.affine)

        peaks_volume = MRIDataVolume(
            peak_dirs, signal.affine)

        seeding_volume = MRIDataVolume(
            seeding.get_fdata(), seeding.affine)
        tracking_volume = MRIDataVolume(
            tracking.get_fdata(), tracking.affine)

        return (signal_volume, peaks_volume, tracking_volume, seeding_volume)

    def get_state_size(self):
        """ Returns the size of the state space by computing the size of
        an example state.

        Returns
        -------
        state_size: int
            Size of the state space.
        """

        example_state = self.reset(0, 1)
        self._state_size = example_state.shape[1]
        return self._state_size

    def get_action_size(self):
        """ Returns the size of the action space.
        """

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

    def _format_actions(
        self,
        actions: np.ndarray,
    ):
        """ Format actions to be used by the environment. Scaling
        actions to the step size.
        """
        actions = normalize_vectors(actions) * self.step_size

        return actions

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

        # Get the last point of each streamline
        segments = streamlines[:, -1, :][:, None, :]

        # Reshape to get a list of coordinates
        N, H, P = segments.shape
        flat_coords = np.reshape(segments, (N * H, P))
        coords = torch.as_tensor(flat_coords).to(self.device)

        # Get the SH coefficients at the last point of each streamline
        # The neighborhood is used to get the SH coefficients around
        # the last point
        signal, _ = interpolate_volume_in_neighborhood(
            self.data_volume,
            coords,
            self.neighborhood_directions)
        N, S = signal.shape

        # Placeholder for the final imputs
        inputs = torch.zeros((N, S + (self.n_dirs * P)), device=self.device)
        # Fill the first part of the inputs with the SH coefficients
        inputs[:, :S] = signal

        # Placeholder for the previous directions
        previous_dirs = np.zeros((N, self.n_dirs, P), dtype=np.float32)
        if L > 1:
            # Compute directions from the streamlines
            dirs = np.diff(streamlines, axis=1)
            # Fetch the N last directions
            previous_dirs[:, :min(dirs.shape[1], self.n_dirs), :] = \
                dirs[:, :-(self.n_dirs+1):-1, :]

        # Flatten the directions to fit in the inputs and send to device
        dir_inputs = torch.reshape(
            torch.from_numpy(previous_dirs).to(self.device),
            (N, self.n_dirs * P))
        # Fill the second part of the inputs with the previous directions
        inputs[:, S:] = dir_inputs

        return inputs

    def _compute_stopping_flags(
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

    def reset(self):
        """ Reset the environment to its initial state.
        """
        if self.compute_reward:
            self.reward_function.reset()

    def step():
        """
        Abstract method to be implemented by subclasses which defines
        the behavior of the environment when taking a step. This includes
        propagating the streamlines, computing the reward, and checking
        which streamlines should stop.
        """
        pass
