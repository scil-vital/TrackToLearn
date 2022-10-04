import nibabel as nib
import numpy as np
import functools
import os

from challenge_scoring.utils.attributes import load_attribs
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from nibabel.streamlines import Tractogram

from TrackToLearn.environments.score import (
    score_tractogram as score, _prepare_gt_bundles_info)
from TrackToLearn.environments.utils import (
    interpolate_volume_at_coordinates,
    is_inside_mask,
    is_too_curvy)
from TrackToLearn.datasets.utils import (
    MRIDataVolume)
from TrackToLearn.utils.utils import (
    normalize_vectors)


class Reward(object):

    def __init__(
        self,
        peaks: MRIDataVolume = None,
        exclude: MRIDataVolume = None,
        target: MRIDataVolume = None,
        max_nb_steps: float = 200,
        max_angle: float = 60,
        min_nb_steps: float = 10,
        asymmetric: bool = False,
        alignment_weighting: float = 1.0,
        straightness_weighting: float = 0.0,
        length_weighting: float = 0.0,
        target_bonus_factor: float = 5.0,
        exclude_penalty_factor: float = 0.0,
        angle_penalty_factor: float = 0.0,
        affine_vox2mask: np.ndarray = None,
        ground_truth_folder: str = None,
        reference: str = None
    ):
        self.peaks = peaks
        self.exclude = exclude
        self.target = target
        self.max_nb_steps = max_nb_steps
        self.max_angle = max_angle
        self.min_nb_steps = min_nb_steps
        self.asymmetric = asymmetric
        self.alignment_weighting = alignment_weighting
        self.straightness_weighting = straightness_weighting
        self.length_weighting = length_weighting
        self.target_bonus_factor = target_bonus_factor
        self.exclude_penalty_factor = exclude_penalty_factor
        self.angle_penalty_factor = angle_penalty_factor
        self.affine_vox2mask = affine_vox2mask
        self.ground_truth_folder = ground_truth_folder
        self.reference = reference

        if self.ground_truth_folder:

            gt_bundles_attribs_path = os.path.join(
                self.ground_truth_folder,
                'gt_bundles_attributes.json')

            basic_bundles_attribs = load_attribs(gt_bundles_attribs_path)

            # Prepare needed scoring data
            masks_dir = os.path.join(self.ground_truth_folder, "masks")
            rois_dir = os.path.join(masks_dir, "rois")
            bundles_dir = os.path.join(self.ground_truth_folder, "bundles")
            bundles_masks_dir = os.path.join(masks_dir, "bundles")
            ref_anat_fname = os.path.join(masks_dir, "wm.nii.gz")

            ROIs = [nib.load(os.path.join(rois_dir, f))
                    for f in sorted(os.listdir(rois_dir))]

            # Get the dict with 'name', 'threshold', 'streamlines',
            # 'cluster_map' and 'mask' for each bundle.
            ref_bundles = _prepare_gt_bundles_info(bundles_dir,
                                                   bundles_masks_dir,
                                                   basic_bundles_attribs,
                                                   ref_anat_fname)

            self.scoring_function = functools.partial(
                score,
                ref_bundles=ref_bundles,
                ROIs=ROIs,
                compute_ic_ib=False)

    def __call__(self, streamlines, dones):
        """
        Compute rewards for the last step of the streamlines
        Each reward component is weighted according to a
        coefficient

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        peaks: `MRIDataVolume`
            Volume containing the fODFs peaks
        target_mask: MRIDataVolume
            Mask representing the tracking endpoints
        exclude_mask: MRIDataVolume
            Mask representing the tracking no-go zones
        max_len: `float`
            Maximum lengths for the streamlines (in terms of points)
        max_angle: `float`
            Maximum degrees between two streamline segments
        alignment_weighting: `float`
            Coefficient for how much reward to give to the alignment
            with peaks
        straightness_weighting: `float`
            Coefficient for how much reward to give to the alignment
            with the last streamline segment
        length_weighting: `float`
            Coefficient for how much to reward the streamline for being
            long
        target_bonus_factor: `float`
            Bonus for streamlines reaching the target mask
        exclude_penalty_factor: `float`
            Penalty for streamlines reaching the exclusion mask
        angle_penalty_factor: `float`
            Penalty for looping or too-curvy streamlines
        affine_vox2mask: np.ndarray
            Affine for moving stuff to voxel space

        Returns
        -------
        rewards: `float`
            Reward components weighted by their coefficients as well
            as the penalites
        """

        N = len(streamlines)

        length = reward_length(streamlines, self.max_nb_steps) \
            if self.length_weighting > 0. else np.zeros((N), dtype=np.uint8)
        alignment = reward_alignment_with_peaks(
            streamlines, self.peaks.data, self.affine_vox2mask,
            self.asymmetric) \
            if self.alignment_weighting > 0 else np.zeros((N), dtype=np.uint8)
        straightness = reward_straightness(streamlines) \
            if self.straightness_weighting > 0 else \
            np.zeros((N), dtype=np.uint8)

        weights = np.asarray([
            self.alignment_weighting, self.straightness_weighting,
            self.length_weighting])
        params = np.stack((alignment, straightness, length))
        rewards = np.dot(params.T, weights)

        # Penalize sharp turns
        if self.angle_penalty_factor > 0.:
            rewards += penalize_sharp_turns(
                streamlines, self.max_angle, self.angle_penalty_factor)

        # Penalize streamlines ending in exclusion mask
        if self.exclude_penalty_factor > 0.:
            rewards += penalize_exclude(
                streamlines,
                self.exclude.data,
                self.affine_vox2mask,
                self.exclude_penalty_factor)

        # Reward streamlines ending in target mask
        if self.target_bonus_factor > 0.:
            rewards += self.reward_target(
                streamlines,
                dones)

        return rewards

    def reward_target(
        self,
        streamlines: np.ndarray,
        dones: np.ndarray,
    ):
        """ Reward streamlines if they end up in the GM

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        target: np.ndarray
            Grey matter mask
        affine_vox2mask : `numpy.ndarray` with shape (4,4) (optional)
            Tranformation that aligns streamlines on top of `mask`.
        penalty_factor: `float`
            Penalty for streamlines ending in target mask
            Should be >= 0

        Returns
        -------
        rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
            Array containing the reward
        """
        # Get boolean array of streamlines ending in mask * penalty
        if streamlines.shape[1] >= self.min_nb_steps and np.any(dones):
            sft = StatefulTractogram(streamlines, self.reference, Space.VOX)
            to_score = np.arange(len(sft))[dones]
            sub_sft = sft[to_score]
            VC, IC, NC = self.scoring_function(sub_sft)

            reward = np.zeros((streamlines.shape[0]))
            if len(VC) > 0:
                reward[to_score[VC]] += self.target_bonus_factor
                self.render(self.peaks, streamlines[to_score[VC]],
                            reward[to_score[VC]])
            if len(IC) > 0:
                reward[to_score[IC]] -= self.target_bonus_factor
            if len(NC) > 0:
                reward[to_score[NC]] -= self.target_bonus_factor
        else:
            reward = np.zeros((streamlines.shape[0]))
        return reward

    def render(
        self,
        peaks,
        streamlines,
        rewards
    ):
        """ Debug function

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
        tractogram = Tractogram(
            streamlines=streamlines,
            data_per_streamline={
                'seeds': streamlines[:, 0, :]
            })

        # Reshape peaks for displaying
        X, Y, Z, M = peaks.data.shape
        peaks = np.reshape(peaks.data, (X, Y, Z, 5, M//5))

        # Setup scene and actors
        scene = window.Scene()

        stream_actor = actor.streamtube(tractogram.streamlines, rewards)
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

        showm = window.ShowManager(scene, reset_camera=True)
        showm.initialize()
        showm.start()


def penalize_exclude(streamlines, exclude, affine_vox2mask, penalty_factor):
    """ Penalize streamlines if they loop

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    exclude: np.ndarray
        CSF matter mask
    affine_vox2mask : `numpy.ndarray` with shape (4,4) (optional)
        Tranformation that aligns streamlines on top of `mask`.
    penalty_factor: `float`
        Penalty for streamlines ending in exclusion mask
        Should be <= 0

    Returns
    -------
    rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array containing the reward
    """
    return (
        is_inside_mask(streamlines, exclude, affine_vox2mask, 0.5) *
        -penalty_factor)


def penalize_sharp_turns(streamlines, max_angle, penalty_factor):
    """ Penalize streamlines if they curve too much

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    max_angle: `float`
        Maximum angle between streamline steps
    penalty_factor: `float`
        Penalty for looping or too-curvy streamlines

    Returns
    -------
    rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array containing the reward
    """
    return is_too_curvy(streamlines, max_angle) * -penalty_factor


def reward_length(streamlines, max_length):
    """ Reward streamlines according to their length w.r.t the maximum length

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space

    Returns
    -------
    rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array containing the reward
    """
    N, S, _ = streamlines.shape

    rewards = np.asarray(
        [streamlines.shape[1]] * len(streamlines)) / max_length

    return rewards


def reward_alignment_with_peaks(
    streamlines, peaks, affine_vox2mask, asymmetric
):
    """ Reward streamlines according to the alignment to their corresponding
        fODFs peaks

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space

    Returns
    -------
    rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array containing the reward
    """
    N, L, _ = streamlines.shape

    if streamlines.shape[1] < 2:
        # Not enough segments to compute curvature
        return np.ones(len(streamlines), dtype=np.uint8)

    X, Y, Z, P = peaks.shape
    idx = streamlines[:, -2].astype(np.int)

    # Get peaks at streamline end
    v = interpolate_volume_at_coordinates(
        peaks, idx, mode='nearest', order=0)
    # Presume 5 peaks (per hemisphere if asymmetric)
    if asymmetric:
        v = np.reshape(v, (N, 5 * 2, P // (5 * 2)))
    else:
        v = np.reshape(v, (N, 5, P // 5))

        with np.errstate(divide='ignore', invalid='ignore'):
            # # Normalize peaks
            v = normalize_vectors(v)

        # Zero NaNs
        v = np.nan_to_num(v)

    # Get last streamline segments

    dirs = np.diff(streamlines, axis=1)
    u = dirs[:, -1]
    # Normalize segments
    with np.errstate(divide='ignore', invalid='ignore'):
        u = normalize_vectors(u)

    # Zero NaNs
    u = np.nan_to_num(u)

    # Get do product between all peaks and last streamline segments
    dot = np.einsum('ijk,ik->ij', v, u)

    if not asymmetric:
        dot = np.abs(dot)

    # Get alignment with the most aligned peak
    rewards = np.amax(dot, axis=-1)
    # rewards = np.abs(dot)

    factors = np.ones((N))

    # Weight alignment with peaks with alignment to itself
    if streamlines.shape[1] >= 3:
        # Get previous to last segment
        w = dirs[:, -2]

        # # Normalize segments
        with np.errstate(divide='ignore', invalid='ignore'):
            w = normalize_vectors(w)

        # # Zero NaNs
        w = np.nan_to_num(w)

        # Calculate alignment between two segments
        np.einsum('ik,ik->i', u, w, out=factors)

    # Penalize angle with last step
    rewards *= factors

    return rewards


def reward_straightness(streamlines):
    """ Reward streamlines according to its sinuosity

    Distance between start and end of streamline / length

    A perfectly straight line has 1.
    A circle would have 0.

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space

    Returns
    -------
    rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array containing the angle between the last two segments
e   """

    N, S, _ = streamlines.shape

    start = streamlines[:, 0]
    end = streamlines[:, -1]

    step_size = 1.
    reward = np.linalg.norm(end - start, axis=1) / (S * step_size)

    return np.clip(reward + 0.5, 0, 1)
