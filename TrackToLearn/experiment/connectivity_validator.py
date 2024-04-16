import itertools

import numpy as np

from dipy.io.streamline import load_tractogram

from scilpy.tractanalysis.reproducibility_measures import compute_dice_voxel
from scilpy.tractanalysis.tools import (
    compute_connectivity, extract_longest_segments_from_profile)

from scilpy.tractograms.uncompress import uncompress

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

        # Filter the streamlines according to their length
        idx_mapping = np.arange(len(sft.streamlines))
        lengths = np.array([len(s) for s in sft.streamlines])
        long_idx = idx_mapping[lengths > 10]

        filt_sft = sft[long_idx]

        # Uncompress the streamlines
        indices, points_to_idx = uncompress(
            filt_sft.streamlines, return_mapping=True)

        con_info = compute_connectivity(indices, data_labels, real_labels,
                                        extract_longest_segments_from_profile)

        comb_list = list(itertools.combinations(real_labels, r=2))
        comb_list.extend(zip(real_labels, real_labels))

        connectivity = np.zeros((len(real_labels), len(real_labels)))
        label_list = real_labels.tolist()
        for in_label, out_label in comb_list:
            pair_info = []
            if in_label not in con_info or out_label not in con_info:
                continue

            if out_label in con_info[in_label]:
                pair_info.extend(con_info[in_label][out_label])

            if in_label in con_info[out_label]:
                pair_info.extend(con_info[out_label][in_label])

            if not len(pair_info):
                continue

            in_pos = label_list.index(in_label)
            out_pos = label_list.index(out_label)

            connectivity[in_pos, out_pos] = len(pair_info)
            connectivity[out_pos, in_pos] = len(pair_info)

        dice, w_dice = compute_dice_voxel(connectivity, env.connectivity)
        corrcoef = np.corrcoef(connectivity.ravel(),
                               env.connectivity.ravel())[0, 1]
        rmse = np.sqrt(np.mean((connectivity - env.connectivity)**2))

        return {'dice': float(dice),
                'w_dice': float(w_dice),
                'corr': float(np.nan_to_num(corrcoef,
                                            nan=0.0)),
                'rmse': rmse,
                'connectivity': (connectivity, env.connectivity)}
