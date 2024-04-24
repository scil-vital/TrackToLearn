import itertools
import numpy as np

from collections import defaultdict
from scipy.ndimage import map_coordinates

from scilpy.tractograms.uncompress import uncompress


class Connectivity():

    def __init__(
        self,
        labels: np.ndarray,
        min_nb_steps: int = 10,
    ):

        self.data_labels = labels
        self.real_labels = np.unique(self.data_labels)[1:]
        self.label_list = self.real_labels.tolist()

        comb_list = list(itertools.combinations(self.real_labels, r=2))
        comb_list.extend(zip(self.real_labels, self.real_labels))

        self.comb_list = comb_list

        self.min_nb_steps = min_nb_steps

    def segmenting_func(
        self, strl_indices, atlas_data, background=0
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
            A list of length 1 with the information dict if ,
            else, an empty list.
        """
        start_label = None
        end_label = None
        start_idx = None
        end_idx = None

        nb_underlying_voxels = len(strl_indices)

        labels = map_coordinates(
            atlas_data, strl_indices.T, order=0, mode='nearest')

        label_idices = np.argwhere(labels != background).squeeze()

        # If the streamline does not traverse any GM voxel, we return an
        # empty list
        # If the streamline is entirely in GM, we return an empty list.
        if (label_idices.size == 1 or len(label_idices) == 0 or
                len(label_idices) == nb_underlying_voxels):
            return []

        start_idx = label_idices[0]
        end_idx = label_idices[-1]

        start_label = labels[start_idx]
        end_label = labels[end_idx]

        return [{'start_label': start_label,
                 'start_index': start_idx,
                 'end_label': end_label,
                 'end_index': end_idx}]

    def compute_connectivity(
        self, indices, atlas_data, real_labels, segmenting_func
    ):
        """
        Parameters
        ----------
        indices: ArraySequence
            The list of 3D indices [i, j, k] of all voxels traversed by all
            streamlines. This is the output of our uncompress function.
        atlas_data: np.ndarray
            The loaded image containing the labels.
        real_labels: list
            The list of labels of interest in the image.
        segmenting_func: Callable
            The function used for segmentation.

        Returns
        -------
        connectivity: dict
            A dict containing one key per real_labels (ex, 1, 2)
            (starting point).
            --The value of connectivity[1] is again a dict with again the
              real_labels as keys.
            --The value of connectivity[1][2] is a list of length n,
            where n is the number of streamlines ending in 1 and finishing
            in 2. Each value is a dict of the following shape:
            {'strl_idx': int  --> The idex of the streamline in the raw data.
             'in_idx: int,    -->
             'out_idx': int}
        """

        def return_labels():
            return {lab: [] for lab in self.real_labels}
        nest = return_labels
        connectivity = defaultdict(nest)

        # toDo. real_labels is not used in segmenting func!
        for strl_idx, strl_vox_indices in enumerate(indices):
            # Managing streamlines out of bound.
            if (np.array(strl_vox_indices) > atlas_data.shape).any():
                continue

            # Finding start_label and end_label.
            segments_info = segmenting_func(strl_vox_indices, atlas_data)
            for si in segments_info:
                connectivity[si['start_label']][si['end_label']].append(
                    {'strl_idx': strl_idx,
                     'in_idx': si['start_index'],
                     'out_idx': si['end_index']})

        return connectivity

    def compute_connectivity_matrix(self, streamlines):
        """ Compute the connectivity matrix from a list of streamlines.

        Parameters
        ----------
        streamlines: list
            A list of streamlines.

        Returns
        -------
        con_info: dict
            The connectivity information.
        """

        # Uncompress the streamlines
        indices = uncompress(
            streamlines, return_mapping=False)

        con_info = self.compute_connectivity(indices,
                                             self.data_labels,
                                             self.real_labels,
                                             self.segmenting_func)

        return con_info
