import nibabel as nib
import numpy as np

from nibabel.streamlines.array_sequence import ArraySequence

from TrackToLearn.experiment.connectivity import Connectivity
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
        dilate: int = 1,
    ):
        # Name for stats
        self.name = 'connectivity_reward'

        self.connectivity = Connectivity(
            labels, min_nb_steps)

        self.min_nb_steps = min_nb_steps

        # Reference connectivity matrix
        self.ref_connectivity = connectivity

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

        con_info = self.connectivity.compute_connectivity_matrix(
            done_streamlines)

        label_list = self.connectivity.label_list
        for in_label, out_label in self.connectivity.comb_list:
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

            if self.ref_connectivity[in_pos, out_pos] > 0:
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
