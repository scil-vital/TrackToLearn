import numpy as np

from dipy.io.streamline import load_tractogram

from scilpy.tractanalysis.reproducibility_measures import compute_dice_voxel

from TrackToLearn.experiment.validators import Validator
from TrackToLearn.experiment.connectivity import Connectivity


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

        connectivity = Connectivity(
            env.labels.data, 0)

        # Load the streamlines
        sft = load_tractogram(filename, env.reference,
                              bbox_valid_check=True, trk_header_check=True)

        sft.to_vox()
        sft.to_corner()

        # Filter the streamlines according to their length
        idx_mapping = np.arange(len(sft.streamlines))
        lengths = np.array([len(s) for s in sft.streamlines])
        long_idx = idx_mapping[lengths > env.min_nb_steps]

        filt_sft = sft[long_idx]

        if len(filt_sft.streamlines) <= 0:
            return {'dice': 0,
                    'w_dice': 0,
                    'corr': 0,
                    'rmse': 0,
                    'connectivity': (
                        np.zeros_like(env.connectivity), env.connectivity)}

        con_info = connectivity.compute_connectivity_matrix(
            filt_sft.streamlines)

        matrix = np.zeros((len(real_labels), len(real_labels)))

        label_list = connectivity.label_list
        for in_label, out_label in connectivity.comb_list:

            in_pos = label_list.index(in_label)
            out_pos = label_list.index(out_label)
            matrix[in_pos, out_pos] += len(con_info[in_label][out_label])
            matrix[out_pos, in_pos] += len(con_info[in_label][out_label])
            matrix[in_pos, out_pos] += len(con_info[out_label][in_label])
            matrix[out_pos, in_pos] += len(con_info[out_label][in_label])

        np.save('connectivity.npy', matrix)

        dice, w_dice = compute_dice_voxel(matrix, env.connectivity)
        corrcoef = np.corrcoef(matrix.ravel(),
                               env.connectivity.ravel())[0, 1]
        rmse = np.sqrt(np.mean((matrix - env.connectivity)**2))

        return {'dice': float(dice),
                'w_dice': float(w_dice),
                'corr': float(np.nan_to_num(corrcoef,
                                            nan=0.0)),
                'rmse': rmse,
                'connectivity': (matrix, env.connectivity)}
