import itertools

import numpy as np

from dipy.io.streamline import load_tractogram
from matplotlib import pyplot as plt

from scilpy.tractanalysis.tools import (
    compute_connectivity, extract_longest_segments_from_profile)
from scilpy.tractanalysis.uncompress import uncompress
from scilpy.tractanalysis.tools import compute_streamline_segment

from TrackToLearn.experiment.validators import Validator


class ConnectivityValidator(Validator):

    def __init__(self):

        self.name = 'Connectivity'

        # Nothing to do before the env is loaded

    def __call__(self, filename, env):
        """ Compute the connectivity matrix from the streamlines and
        the labels from the env' subject. Compare it to the reference
        connectivity matrix by computing the correlation between the
        two matrices.
        """

        data_labels = env.labels.data
        real_labels = np.unique(data_labels)[1:]   # Removing the background 0.

        # Load the streamlines
        sft = load_tractogram(filename, 'same', bbox_valid_check=False)

        sft.to_vox()
        sft.to_corner()

        # Uncompress the streamlines
        indices, points_to_idx = uncompress(
            sft.streamlines, return_mapping=True)

        con_info = compute_connectivity(indices, data_labels, real_labels,
                                        extract_longest_segments_from_profile)

        comb_list = list(itertools.combinations(real_labels, r=2))
        comb_list.extend(zip(real_labels, real_labels))


        connectivity = np.zeros((len(real_labels), len(real_labels)))

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

            connectivity[int(in_label), int(out_label)] = len(pair_info)

        # Normalize the connectivity matrix
        connectivity = connectivity / np.max(connectivity)

        # Display the connectivity matrix using matplotlib
        plt.imshow(connectivity, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()

        # Compute the correlation between the reference and the computed
        # connectivity matrix
        correlation = np.corrcoef(env.connectivity.flatten(),
                                    connectivity.flatten())[0, 1]

        return {'corr': correlation}
