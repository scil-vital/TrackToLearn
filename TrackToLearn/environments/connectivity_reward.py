import itertools
import nibabel as nib
import numpy as np

from nibabel.streamlines.array_sequence import ArraySequence

from scilpy.tractanalysis.tools import (
    compute_connectivity, extract_longest_segments_from_profile)
from scilpy.tractograms.uncompress import uncompress

from TrackToLearn.environments.reward import Reward


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
    ):
        # Name for stats
        self.name = 'connectivity_reward'

        self.labels = labels
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

        data_labels = self.labels
        real_labels = np.unique(data_labels)[1:]   # Removing the background 0.

        # Uncompress the streamlines
        indices = uncompress(
            streamlines, return_mapping=False)

        con_info = compute_connectivity(indices, data_labels, real_labels,
                                        extract_longest_segments_from_profile)

        comb_list = list(itertools.combinations(real_labels, r=2))
        comb_list.extend(zip(real_labels, real_labels))

        label_list = real_labels.tolist()
        for in_label, out_label in comb_list:
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
                    reward[strl_idx] = 1

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
