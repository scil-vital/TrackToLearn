import itertools
import nibabel as nib
import numpy as np

from scipy.ndimage import map_coordinates
from nibabel.streamlines.array_sequence import ArraySequence

from scilpy.image.labels import dilate_labels
from scilpy.tractanalysis.tools import (
    compute_connectivity)
from scilpy.tractograms.uncompress import uncompress

from TrackToLearn.environments.reward import Reward


def extract_longest_segments_from_profile(
    strl_indices, atlas_data, background=0
):
    """
    For one given streamline, find the labels at both ends.

    Parameters
    ----------
    strl_indices: np.ndarray
        The indices of all voxels traversed by this streamline.
    atlas_data: np.ndarray
        The loaded image containing the labels.
    background: int
        The value of the background in the atlas.

    Returns
    -------
    segments_info: list[dict]
        A list of length 1 with the information dict if , else, an empty list.
    """
    start_label = None
    end_label = None
    start_idx = None
    end_idx = None

    nb_underlying_voxels = len(strl_indices)

    labels = map_coordinates(
        atlas_data, strl_indices.T, order=0, mode='nearest')

    label_idices = np.argwhere(labels != background).squeeze()

    # If the streamline does not traverse any GM voxel, we return an empty list
    # If the streamline is entirely in GM, we return an empty list.
    if label_idices.size == 1 or len(label_idices) == nb_underlying_voxels:
        return []

    start_idx = label_idices[0]
    end_idx = label_idices[-1]

    start_label = labels[start_idx]
    end_label = labels[end_idx]

    return [{'start_label': start_label,
             'start_index': start_idx,
             'end_label': end_label,
             'end_index': end_idx}]


class ConnectivityReward(Reward):

    """ Reward streamlines based on the predicted scores of an "Oracle".
    A binary reward is given by the oracle at the end of tracking.
    """

    def __init__(
        self,
        labels: np.ndarray,
        connectivity: np.ndarray,
        reference: nib.Nifti1Image,
        affine_vox2rasmm: np.ndarray,
        min_nb_steps: int = 10,
        dilate: int = 2,
    ):
        # Name for stats
        self.name = 'connectivity_reward'

        if dilate > 0:
            diag = np.diag_indices(4)
            vox_size = np.mean(affine_vox2rasmm[diag][:3])
            distance = vox_size * dilate
            labels = dilate_labels(labels, vox_size, distance, 1,
                                   labels_not_to_dilate=[],
                                   labels_to_fill=[0])

        self.data_labels = labels
        self.real_labels = np.unique(self.data_labels)[1:]

        comb_list = list(itertools.combinations(self.real_labels, r=2))
        comb_list.extend(zip(self.real_labels, self.real_labels))

        self.comb_list = comb_list

        self.min_nb_steps = min_nb_steps

        # Reference connectivity matrix
        self.connectivity = connectivity

        # Reference anatomy
        self.reference = reference

        # Affine matrix from voxel to rasmm
        self.affine_vox2rasmm = affine_vox2rasmm

    def reward(self, streamlines, dones):
        """ Compute the reward for each streamline by comparing the
        connectivity of the streamlines to the ground truth labels.
        Reward streamlines if they connect two regions that are
        connected in the ground truth labels.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space

        Returns
        -------
        rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array containing the reward
        """

        reward = np.zeros((len(streamlines)))
        all_idx = np.arange(len(streamlines))
        done_streamlines = streamlines[dones.astype(bool)]

        # Uncompress the streamlines
        indices = uncompress(
            done_streamlines, return_mapping=False)

        con_info = compute_connectivity(indices,
                                        self.data_labels, self.real_labels,
                                        extract_longest_segments_from_profile)

        label_list = self.real_labels.tolist()
        for in_label, out_label in self.comb_list:
            pair_info = []
            if in_label not in con_info:
                continue
            elif out_label in con_info[in_label]:
                pair_info.extend(con_info[in_label][out_label])
                if out_label not in con_info:
                    continue
                elif in_label in con_info[out_label]:
                    pair_info.extend(con_info[out_label][in_label])
                    if not len(pair_info):
                        continue

            in_pos = label_list.index(in_label)
            out_pos = label_list.index(out_label)

            if self.connectivity[in_pos, out_pos] > 0:
                for connection in pair_info:
                    strl_idx = connection['strl_idx']
                    actual_idx = all_idx[dones.astype(bool)][strl_idx]
                    reward[actual_idx] = 1

        return reward * dones

    def __call__(
        self,
        streamlines: np.ndarray,
        dones: np.ndarray,
    ):

        N, L, P = streamlines.shape

        if L > self.min_nb_steps and sum(dones.astype(int)) > 0:
            array_seq = ArraySequence(streamlines)
            return self.reward(array_seq, dones)
        return np.zeros((N))

    def visualize_connection(self, streamline, labels, affine_vox2rasmm):
        """ Visualize the connection of a streamline by displaying it
        as a tube alongside the ROIs it connects.
        """

        from dipy.viz import window, actor

        # Create a new scene
        scene = window.Scene()

        # Add the streamline
        scene.add(actor.line(streamline, linewidth=0.1))

        # Add a slice of the labels
        scene.add(actor.slicer(labels, affine=affine_vox2rasmm))

        # Display the scene
        window.show(scene)
