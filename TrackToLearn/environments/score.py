#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import os

import nibabel as nib
import numpy as np

from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import set_number_of_points
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.metrics import length as slength

from challenge_scoring import NB_POINTS_RESAMPLE
from challenge_scoring.metrics.invalid_connections import group_and_assign_ibs
from challenge_scoring.metrics.valid_connections import auto_extract_VCs


def _prepare_gt_bundles_info(bundles_dir, bundles_masks_dir,
                             gt_bundles_attribs, ref_anat_fname):
    """
    Returns
    -------
    ref_bundles: list[dict]
        Each dict will contain {'name': 'name_of_the_bundle',
                                'threshold': thres_value,
                                'streamlines': list_of_streamlines},
                                'cluster_map': the qb cluster map,
                                'mask': the loaded bundle mask (nifti).}
    """
    qb = QuickBundles(20, metric=AveragePointwiseEuclideanMetric())

    ref_bundles = []

    for bundle_idx, bundle_f in enumerate(sorted(os.listdir(bundles_dir))):
        bundle_name = os.path.splitext(os.path.basename(bundle_f))[0]

        bundle_attribs = gt_bundles_attribs.get(os.path.basename(bundle_f))
        if bundle_attribs is None:
            raise ValueError(
                "Missing basic bundle attribs for {0}".format(bundle_f))

        orig_sft = load_tractogram(
            os.path.join(bundles_dir, bundle_f), ref_anat_fname,
            bbox_valid_check=False, trk_header_check=False)
        orig_sft.to_vox()
        orig_sft.to_center()

        # Already resample to avoid doing it for each iteration of chunking
        orig_strl = orig_sft.streamlines

        resamp_bundle = set_number_of_points(orig_strl, NB_POINTS_RESAMPLE)
        resamp_bundle = [s.astype(np.float32) for s in resamp_bundle]

        bundle_cluster_map = qb.cluster(resamp_bundle)
        bundle_cluster_map.refdata = resamp_bundle

        bundle_mask = nib.load(os.path.join(bundles_masks_dir,
                                            bundle_name + '.nii.gz'))

        ref_bundles.append({'name': bundle_name,
                            'threshold': bundle_attribs['cluster_threshold'],
                            'cluster_map': bundle_cluster_map,
                            'mask': bundle_mask})

    return ref_bundles


def score_tractogram(sft,
                     ref_bundles=None,
                     ROIs=None,
                     compute_ic_ib: bool = False):
    """
    Score a submission, using the following algorithm:
        1: extract all streamlines that are valid, which are classified as
           Valid Connections (VC) making up Valid Bundles (VB).
        2: remove streamlines shorter than an threshold based on the GT dataset
        3: cluster the remaining streamlines
        4: remove singletons
        5: assign each cluster to the closest ROIs pair. Those make up the
           Invalid Connections (IC), grouped as Invalid Bundles (IB).
        6: streamlines that are neither in VC nor IC are classified as
           No Connection (NC).


    Parameters
    ------------
    streamlines_fname : string
        path to the file containing the streamlines.
    base_data_dir : string
        path to the direction containing the scoring data.
    basic_bundles_attribs : dictionary
        contains the attributes of the basic bundles
        (name, list of streamlines, segmentation threshold)
    save_full_vc : bool
        indicates if the full set of VC will be saved in an individual file.
    save_full_ic : bool
        indicates if the full set of IC will be saved in an individual file.
    save_full_nc : bool
        indicates if the full set of NC will be saved in an individual file.
    compute_ic_ib:
        segment IC results into IB.
    save_IBs : bool
        indicates if the invalid bundles will be saved in individual file for
        each IB.
    save_VBs : bool
        indicates if the valid bundles will be saved in individual file for
        each VB.
    segmented_out_dir : string
        the path to the directory where segmented files will be saved.
    segmented_base_name : string
        the base name to use for saving segmented files.
    out_tract_type: str
        extension for the output tractograms.
    verbose : bool
        indicates if the algorithm needs to be verbose when logging messages.

    Returns
    ---------
    scores : dict
        dictionnary containing a score for each metric
    """

    sft.to_vox()
    sft.to_center()
    total_strl_count = len(sft.streamlines)

    # Extract VCs and VBs, compute OL, OR, f1 for each.
    VC_indices, found_vbs_info = auto_extract_VCs(sft, ref_bundles)
    VC = np.asarray(VC_indices, dtype=np.int32)

    candidate_ic_strl_indices = np.setdiff1d(range(total_strl_count),
                                             VC_indices)
    if compute_ic_ib:

        candidate_ic_indices = []
        rejected_indices = []

        # Chosen from GT dataset
        length_thres = 35.

        # Filter streamlines that are too short, consider them as NC
        for idx in candidate_ic_strl_indices:
            if slength(sft.streamlines[idx]) >= length_thres:
                candidate_ic_indices.append(idx)
            else:
                rejected_indices.append(idx)

        ic_counts = 0
        nb_ib = 0

        if len(candidate_ic_indices):
            additional_rejected_indices, ic_counts, nb_ib = \
                group_and_assign_ibs(sft, candidate_ic_indices,  ROIs,
                                     False, False, '',
                                     '', '',
                                     'tck')

            rejected_indices.extend(additional_rejected_indices)

        if ic_counts != len(candidate_ic_strl_indices) - len(rejected_indices):
            raise ValueError("Some streamlines were not correctly assigned to "
                             "NC")

        IC = candidate_ic_strl_indices
    else:
        IC = []
        rejected_indices = candidate_ic_strl_indices

    # Converting np.float to floats for json dumps
    NC = rejected_indices

    return VC, IC, NC
