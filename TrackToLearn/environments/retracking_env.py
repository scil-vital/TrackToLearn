import numpy as np

from nibabel.streamlines import Tractogram
from typing import Tuple


from TrackToLearn.environments.tracking_env import TrackingEnvironment


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
        self.theta = env_dto['theta']
        self.epsilon = env_dto['epsilon']

        self.npv = env_dto['npv']
        self.cmc = env_dto['cmc']
        self.binary_stopping_threshold = env_dto['binary_stopping_threshold']
        self.asymmetric = env_dto['asymmetric']

        self.action_type = env_dto['action_type']

        self.sphere = env.sphere

        self.oracle_checkpoint = env_dto['oracle_checkpoint']

        self.oracle_stopping_criterion = env_dto['oracle_stopping_criterion']
        self.oracle_filter = False

        self.rng = env_dto['rng']
        self.device = env_dto['device']

        self.seeding_data = env.seeding_data

        self.step_size = env.step_size
        self.min_length = env.min_length
        self.max_length = env.max_length

        # Compute maximum length
        self.max_nb_steps = env.max_nb_steps
        self.min_nb_steps = env.min_nb_steps

        # Neighborhood used as part of the state
        self.add_neighborhood_vox = env.add_neighborhood_vox
        self.neighborhood_directions = env.neighborhood_directions

        self.compute_reward = env.compute_reward
        self.reward_function = env.reward_function
        self.stopping_criteria = env.stopping_criteria

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

    def reset(self, half_tractogram: Tractogram) -> np.ndarray:
        """ Initialize tracking from half-streamlines.

        Parameters
        ----------
        half_tractogram: Tractogram
            Half-streamlines to initialize environment, in RASMM space.

        Returns
        -------
        state: numpy.ndarray
            Initial state for RL model
        """

        # super().reset()

        # Bring streamlines back into vox space
        half_tractogram.apply_affine(self.affine_rasmm2vox)
        # Get half-streamlines
        half_streamlines = half_tractogram.streamlines

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

        # Initialize rewards and done flags
        self.flags = np.zeros(N, dtype=int)
        self.dones = np.full(N, False)
        self.continue_idx = np.arange(N)
        self.state = self._format_state(
            self.streamlines[self.continue_idx, :self.length])

        # Setup input signal
        return self.state[self.continue_idx]

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, list, bool, dict]:
        """
        Apply actions and grow streamlines for one step forward
        Calculate rewards and if the tracking is done. While tracking
        has not surpassed half-streamlines, replace the tracking step
        taken with the actual streamline segment.

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

        directions = self._format_actions(actions)

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
        self.not_stopping = np.logical_not(stopping)

        self.new_continue_idx, self.stopping_idx = (
            self.continue_idx[~stopping],
            self.continue_idx[stopping])

        # Set "done" flags for RL
        self.dones[self.stopping_idx] = 1

        # Keep the reason why tracking stopped
        self.flags[
            self.stopping_idx] = new_flags[stopping]

        reward = np.zeros(self.streamlines.shape[0])
        reward_info = {}
        # Compute reward if wanted. At valid time, no need
        # to compute it and slow down the tracking process
        if self.compute_reward:
            reward, reward_info = self.reward_function(
                self.streamlines[self.continue_idx, :self.length],
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
            {'continue_idx': self.continue_idx,
             'reward_info': reward_info})
