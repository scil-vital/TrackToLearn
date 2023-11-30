import functools
from typing import Callable, Dict, Tuple

import h5py
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
from gymnasium.wrappers.normalize import RunningMeanStd
from nibabel.streamlines import Tractogram
from scilpy.reconst.utils import (find_order_from_nb_coeff, get_b_matrix,
                                  get_maximas)

from TrackToLearn.datasets.utils import (MRIDataVolume, SubjectData,
                                         convert_length_mm2vox,
                                         set_sh_order_basis)
from TrackToLearn.environments.coverage_reward import CoverageReward
from TrackToLearn.environments.filters import (CMCFilter, Filters,
                                               MinLengthFilter, OracleFilter)
from TrackToLearn.environments.local_reward import (LengthReward,
                                                    PeaksAlignmentReward,
                                                    TargetReward)
from TrackToLearn.environments.oracle_reward import OracleReward
from TrackToLearn.environments.reward import RewardFunction
from TrackToLearn.environments.tractometer_reward import TractometerReward
from TrackToLearn.environments.stopping_criteria import (
    BinaryStoppingCriterion, CmcStoppingCriterion, OracleStoppingCriterion,
    StoppingFlags)
from TrackToLearn.environments.utils import (  # is_looping,
    is_too_curvy, is_too_long)
from TrackToLearn.utils.utils import from_polar, from_sphere, normalize_vectors

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

        # TODO: Split this into several functions, which can
        # be reused in `set_step_size`

        # Reference file
        self.reference = env_dto['reference']

        # Affines
        self.affine_vox2rasmm = input_volume.affine_vox2rasmm
        self.affine_rasmm2vox = np.linalg.inv(self.affine_vox2rasmm)

        # Volumes and masks
        self.data_volume = torch.from_numpy(
            input_volume.data).to(env_dto['device'], dtype=torch.float32)
        self.tracking_mask = tracking_mask
        self.target_mask = target_mask
        self.include_mask = include_mask
        self.exclude_mask = exclude_mask
        self.peaks = peaks
        mask_data = tracking_mask.data.astype(np.uint8)
        self.seeding_data = seeding_mask.data.astype(np.uint8)

        # Unused: this is from an attempt to normalize the input data
        # as is done by the original PPO impl
        # Does not seem to be necessary here.
        self.normalize_obs = False  # env_dto['normalize']
        self.obs_rms = None

        self._state_size = None  # to be calculated later

        # Tracking parameters
        self.n_signal = env_dto['n_signal']
        self.n_dirs = env_dto['n_dirs']
        self.theta = theta = env_dto['theta']
        self.epsilon = env_dto['epsilon']
        # Number of seeds per voxel
        self.npv = env_dto['npv']
        # Whether to use CMC or binary stopping criterion
        self.cmc = env_dto['cmc']
        self.binary_stopping_threshold = env_dto['binary_stopping_threshold']
        # Whether to use asymmetric fODFs. Has not been tested in
        # a while.
        self.asymmetric = env_dto['asymmetric']
        # Action type (discrete, polar, cartesian)
        # The code for polar and discrete actions is not finished.
        self.action_type = env_dto['action_type']

        if env_dto['sphere']:
            self.sphere = get_sphere(env_dto['sphere'])
        else:
            self.sphere = None

        # Step-size and min/max lengths are typically defined in mm
        # by the user, but need to be converted to voxels.
        step_size_mm = env_dto['step_size']
        min_length_mm = env_dto['min_length']
        max_length_mm = env_dto['max_length']
        add_neighborhood_mm = env_dto['add_neighborhood']

        # Oracle parameters
        self.oracle_checkpoint = env_dto['oracle_checkpoint']
        self.oracle_stopping_criterion = env_dto['oracle_stopping_criterion']
        self.oracle_filter = env_dto['oracle_filter']

        # Tractometer parameters
        self.tractometer_weighting = env_dto['tractometer_weighting']
        self.scoring_data = env_dto['scoring_data']

        # Other parameters
        self.rng = env_dto['rng']
        self.device = env_dto['device']

        self.step_size = convert_length_mm2vox(
            step_size_mm,
            self.affine_vox2rasmm)
        self.min_length = min_length_mm
        self.max_length = max_length_mm

        # Compute maximum length
        self.max_nb_steps = int(self.max_length / step_size_mm)
        self.min_nb_steps = int(self.min_length / step_size_mm)

        # Neighborhood used as part of the state
        self.add_neighborhood_vox = None
        if add_neighborhood_mm:
            # TODO: This is a hack. The neighborhood should be computed
            # from the step size, not from a separate parameter.
            self.add_neighborhood_vox = convert_length_mm2vox(
                add_neighborhood_mm,
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
        print(
            '{} has {} seeds.'.format(self.__class__.__name__,
                                      len(self.seeds)))

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
            functools.partial(is_too_curvy, max_theta=theta)

        # Streamline loop criterion (not used, too slow)
        # self.stopping_criteria[
        #     StoppingFlags.STOPPING_LOOP] = \
        #     functools.partial(is_looping,
        #                       loop_threshold=360)

        # Angle between peaks and segments (angular error criterion)
        # Not used a it constrains tracking too much.
        # self.stopping_criteria[
        #     StoppingFlags.STOPPING_ANGULAR_ERROR] = AngularErrorCriterion(
        #     self.epsilon,
        #     self.peaks)

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
                self.binary_stopping_threshold)
            self.stopping_criteria[StoppingFlags.STOPPING_MASK] = \
                binary_criterion

        # ==========================================
        # Reward function
        # =========================================

        # Reward parameters
        self.compute_reward = env_dto['compute_reward']
        # "Local" reward parameters
        self.alignment_weighting = env_dto['alignment_weighting']
        self.straightness_weighting = env_dto['straightness_weighting']
        self.length_weighting = env_dto['length_weighting']
        self.target_bonus_factor = env_dto['target_bonus_factor']
        self.exclude_penalty_factor = env_dto['exclude_penalty_factor']
        self.angle_penalty_factor = env_dto['angle_penalty_factor']
        self.coverage_weighting = env_dto['coverage_weighting']
        self.tractometer_weighting = env_dto['tractometer_weighting']

        # Oracle reward parameters
        self.dense_oracle = env_dto['dense_oracle_weighting'] > 0
        if self.dense_oracle:
            self.oracle_weighting = env_dto['dense_oracle_weighting']
        else:
            self.oracle_weighting = env_dto['sparse_oracle_weighting']

        # Reward function and reward factors
        if self.compute_reward:
            # Reward streamline according to alignment with local peaks
            peaks_reward = PeaksAlignmentReward(self.peaks, self.asymmetric)
            # Reward streamlines if they reach a specific mask
            # (i.e. grey matter)
            target_reward = TargetReward(self.target_mask)
            # Reward streamlines for their length
            length_reward = LengthReward(self.max_nb_steps)
            # Reward streamlines according to an oracle
            oracle_reward = OracleReward(self.oracle_checkpoint,
                                         self.dense_oracle,
                                         self.reference,
                                         self.affine_vox2rasmm,
                                         self.device)
            # Reward streamlines according to tractometer
            # This should not be used
            tractometer_reward = TractometerReward(self.scoring_data,
                                                   self.reference,
                                                   self.affine_vox2rasmm)

            # Reward streamlines according to coverage of the WM mask.
            cover_reward = CoverageReward(self.tracking_mask)
            # Combine all reward factors into the reward function
            # TODO: There has to be a better way of declaring the reward
            # function as now the resp. of not crashing if some stuff is
            # missing (i.e. the target mask) is on the reward factors.
            # They should just not be initialized instead.
            self.reward_function = RewardFunction(
                [peaks_reward,
                 target_reward,
                 length_reward,
                 oracle_reward,
                 tractometer_reward,
                 cover_reward],
                [self.alignment_weighting,
                 self.target_bonus_factor,
                 self.length_weighting,
                 self.oracle_weighting,
                 self.tractometer_weighting,
                 self.coverage_weighting])

        # ==========================================
        # Filters
        # =========================================
        # Filters are applied when calling env.get_streamlines()
        # They are used to filter out streamlines that do not meet
        # certain criteria (e.g. length, oracle, CMC, etc.)

        self.filters = {}
        # Filter out streamlines below the length threshold
        # self.filters[Filters.MIN_LENGTH] = MinLengthFilter(self.min_nb_steps)

        # Filter out streamlines according to the oracle
        if self.oracle_filter:
            self.filters[Filters.ORACLE] = OracleFilter(self.oracle_checkpoint,
                                                        self.min_nb_steps,
                                                        self.reference,
                                                        self.affine_vox2rasmm,
                                                        self.device)

        # Filter out streamlines according to the Continuous Map Criterion
        if self.cmc:
            self.filters[Filters.CMC] = CMCFilter(self.include_mask.data,
                                                  self.exclude_mask.data,
                                                  self.affine_vox2rasmm,
                                                  self.step_size,
                                                  self.min_nb_steps)

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
        wm_file = env_dto['wm_file']
        in_seed = env_dto['in_seed']
        in_mask = env_dto['in_mask']
        sh_basis = env_dto['sh_basis']

        input_volume, peaks_volume, tracking_mask, seeding_mask = \
            BaseEnv._load_files(
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
            peaks_volume,
            env_dto)

    @classmethod
    def _load_dataset(
        cls, dataset_file, split_id, subject_id, interface_seeding=False
    ):
        """ Load data volumes and masks from the HDF5

        Should everything be put into `self` ? Should everything be returned
        instead ?

        Parameters
        ----------
        dataset_file: str
            Path to the HDF5 file containing the dataset.
        split_id: str
            Name of the split to load (e.g. 'training',
            'validation', 'testing').
        subject_id: str
            Name of the subject to load (e.g. '100307')
        interface_seeding: bool
            Whether to seed from the interface or from the WM.

        Returns
        -------
        input_volume: MRIDataVolume
            Volumetric data containing the SH coefficients
        tracking_mask: MRIDataVolume
            Volumetric mask where tracking is allowed
        include_mask: MRIDataVolume
            Mask representing the tracking go zones. Only useful if
            using CMC.
        exclude_mask: MRIDataVolume
            Mask representing the tracking no-go zones. Only useful if
            using CMC.
        target_mask: MRIDataVolume
            Mask representing the tracking endpoints
        seeding_mask: MRIDataVolume
            Mask where seeding should be done
        peaks: MRIDataVolume
            Volume containing the fODFs peaks
        """

        print("Loading {} from the {} set.".format(subject_id, split_id))
        # Load input volume
        with h5py.File(
                dataset_file, 'r'
        ) as hdf_file:
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
        """ Load data volumes and masks from files. This is useful for
        tracking from a trained model.

        If the signal is not in descoteaux07 basis, it will be converted. The
        WM mask will be loaded and concatenated to the signal. Additionally,
        peaks will be computed from the signal.

        Parameters
        ----------
        signal_file: str
            Path to the signal file (e.g. SH coefficients).
        wm_file: str
            Path to the WM mask file.
        in_seed: str
            Path to the seeding mask file.
        in_mask: str
            Path to the tracking mask file.
        sh_basis: str
            SH basis of the signal file.

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
                                  target_order=6,
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
        wm = nib.load(wm_file)
        wm_data = wm.get_fdata()
        if len(wm_data.shape) == 3:
            wm_data = wm_data[..., None]

        signal_data = np.concatenate(
            [data, wm_data], axis=-1)
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
        """ Returns the size of the action space. This depends on the
        action type. If the action type is discrete, the action space is
        the number of vertices on the sphere. If the action type is polar,
        the action space is 2 (theta and phi). If the action type is
        cartesian, the action space is 3 (x, y, z).
        """

        if self.action_type == 'discrete':
            return len(self.sphere.vertices)
        elif self.action_type == 'polar':
            return 2
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

        TODO: Code should be reused from `__init__` instead of copy-pasted.
        """

        self.step_size = convert_length_mm2vox(
            step_size_mm,
            self.affine_vox2rasmm)

        if self.add_neighborhood_vox:
            self.add_neighborhood_vox = convert_length_mm2vox(
                step_size_mm,
                self.affine_vox2rasmm)
            self.neighborhood_directions = torch.cat(
                (torch.zeros((1, 3)),
                 get_neighborhood_vectors_axes(1, self.add_neighborhood_vox))
            ).to(self.device)

        # Compute maximum length
        self.max_nb_steps = int(self.max_length / step_size_mm)
        self.min_nb_steps = int(self.min_length / step_size_mm)

        if self.compute_reward:
            peaks_reward = PeaksAlignmentReward(self.peaks, self.asymmetric)
            target_reward = TargetReward(self.target_mask)
            length_reward = LengthReward(self.max_nb_steps)
            oracle_reward = OracleReward(self.oracle_checkpoint,
                                         self.reference,
                                         self.affine_vox2rasmm,
                                         self.device)
            cover_reward = CoverageReward(self.tracking_mask)
            self.reward_function = RewardFunction(
                [peaks_reward, target_reward,
                 length_reward, oracle_reward,
                 cover_reward],
                [self.alignment_weighting,
                 self.target_bonus_factor,
                 self.length_weighting,
                 self.oracle_weighting,
                 self.coverage_weighting])

        self.stopping_criteria[StoppingFlags.STOPPING_LENGTH] = \
            functools.partial(is_too_long,
                              max_nb_steps=self.max_nb_steps)

        self.stopping_criteria[
            StoppingFlags.STOPPING_ANGULAR_ERROR] = OracleStoppingCriterion(
            self.oracle_checkpoint,
            self.min_nb_steps * 2,
            self.reference,
            self.affine_vox2rasmm,
            self.device)

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

    def _format_actions(
        self,
        actions: np.ndarray,
    ):
        """ Format actions to be used by the environment. This includes
        converting spherical actions to cartesian actions, and scaling
        actions to the step size.

        This is a leftover from an attempt to use spherical and discrete
        actions.
        """

        if self.action_type == 'polar' and actions.shape[-1] == 2:
            actions = from_polar(actions)
        if self.action_type == 'discrete' and actions.shape[-1] != 3:
            actions = from_sphere(actions, self.sphere)
        # Scale actions to step size
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

        from fury import actor, window

        # Might be rendering from outside the environment
        if tractogram is None:

            tractogram = self.get_streamlines()

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
        scene.add(stream_actor)
        scene.add(peak_actor)
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
