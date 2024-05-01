import itertools
import numpy as np

from collections import defaultdict
from scipy.ndimage import map_coordinates


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
        self, streamlines, atlas_data, background=0
    ):
        """
        For one given streamline, find the labels at both ends.

        Parameters
        ----------
        streamlines: np.ndarray
            The streamlines to be processed.
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

        start_voxels = np.asarray([s[0] for s in streamlines])

        start_labels = map_coordinates(
            atlas_data, start_voxels.T, order=0, mode='nearest')

        end_voxels = np.asarray([s[-1] for s in streamlines])

        end_labels = map_coordinates(
            atlas_data, end_voxels.T, order=0, mode='nearest')

        return start_labels, end_labels

    def compute_connectivity(
        self, streamlines, atlas_data, real_labels, segmenting_func
    ):
        """
        Parameters
        ----------
        streamlines: np.ndarray
            The streamlines to be processed.
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

        start_labels, end_labels = segmenting_func(streamlines, atlas_data)

        # toDo. real_labels is not used in segmenting func!
        for strl_idx, start_label, end_label in enumerate(
            zip(start_labels, end_labels)
        ):
            if start_label == 0 or end_label == 0:
                continue
            connectivity[start_label][end_label].append(
                {'strl_idx': strl_idx,
                 'in_idx': 0,
                 'out_idx': len(streamlines[strl_idx]) - 1})

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

        con_info = self.compute_connectivity(streamlines,
                                             self.data_labels,
                                             self.real_labels,
                                             self.segmenting_func)

        return con_info
