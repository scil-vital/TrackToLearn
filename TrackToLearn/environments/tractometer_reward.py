
import itertools
import json
import logging
import os
import tempfile
from collections import namedtuple

import nibabel as nib
import numpy as np
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Tractogram
from scipy.ndimage import binary_dilation

from scilpy.io.image import get_data_as_mask
from scilpy.segment.streamlines import filter_grid_roi, filter_grid_roi_both
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.filenames import split_name_with_nii

from TrackToLearn.environments.reward import Reward

def_len = [0, np.inf]


def load_and_verify_everything(
    reference,
    gt_config,
    gt_dir,
    use_gt_masks_as_all_masks,
    args,
):
    """
    - Reads the config file
    - Loads the masks / sft
        - If endpoints were given instead of head + tail, separate into two
          sub-rois.
    - Verifies compatibility
    """

    # Read the config file
    (bundle_names, gt_masks_files, all_masks_files, any_masks_files,
     roi_options, lengths, angles, orientation_lengths,
     abs_orientation_lengths) = read_config_file(
         gt_config, gt_dir, use_gt_masks_as_all_masks)

    # Find every mandatory mask to be loaded
    list_masks_files_r = list(itertools.chain(
        *[list(roi_option.values()) for roi_option in roi_options]))
    list_masks_files_o = gt_masks_files + all_masks_files + any_masks_files
    # (This removes duplicates:)
    list_masks_files_r = list(dict.fromkeys(list_masks_files_r))
    list_masks_files_o = list(dict.fromkeys(list_masks_files_o))

    logging.info("Loading and/or computing ground-truth masks, limits "
                 "masks and any_masks.")
    gt_masks = compute_masks_from_bundles(gt_masks_files, reference)
    inv_all_masks = compute_masks_from_bundles(all_masks_files, reference,
                                               inverse_mask=True)
    any_masks = compute_masks_from_bundles(any_masks_files, reference)

    logging.info("Extracting ground-truth head and tail masks.")
    gt_tails, gt_heads = compute_endpoint_masks(roi_options, args)

    # Update the list of every ROI, remove duplicates
    list_rois = gt_tails + gt_heads

    return (gt_tails, gt_heads, bundle_names, list_rois,
            lengths, angles, orientation_lengths, abs_orientation_lengths,
            inv_all_masks, gt_masks, any_masks)


def read_config_file(
    gt_config, gt_dir='', use_gt_masks_as_all_masks=False
):
    """
    Reads the gt_config file and returns:

    Returns
    -------
    bundles: List
        The names of each bundle.
    gt_masks: List
        The gt_mask filenames per bundle (None if not set) (used for
        tractometry statistics).
    all_masks: List
        The all_masks filenames per bundles (None if not set).
    any_masks: List
        The any_masks filenames per bundles (None if not set).
    roi_options: List
        The roi_option dict per bundle. Keys are 'gt_head', 'gt_tail' if
        they are set, else 'gt_endpoints'.
    angles: List
        The maximum angles per bundle (None if not set).
    lengths: List
        The [min max] lengths per bundle (None if not set).
    orientation_length: List
        The [[min_x, max_x], [min_y, max_y], [min_z, max_z]] per bundle.
        (None they are all not set).
    """
    angles = []
    lengths = []
    orientation_lengths = []
    abs_orientation_lengths = []
    gt_masks = []
    all_masks = []
    any_masks = []
    roi_options = []
    show_warning_gt = False

    with open(gt_config, "r") as json_file:
        config = json.load(json_file)

        bundles = list(config.keys())
        for bundle in bundles:
            bundle_config = config[bundle]

            if 'gt_mask' not in bundle_config:
                show_warning_gt = True
            if 'endpoints' not in bundle_config and \
                    'head' not in bundle_config:
                raise ValueError(
                    "Bundle configuration for bundle {} misses 'endpoints' or "
                    "'head'/'tail'".format(bundle))

            angle = length = None
            length_x = length_y = length_z = None
            length_x_abs = length_y_abs = length_z_abs = None
            gt_mask = all_mask = any_mask = roi_option = None

            for key in bundle_config.keys():
                if key == 'angle':
                    angle = bundle_config['angle']
                elif key == 'length':
                    length = bundle_config['length']
                elif key == 'length_x':
                    length_x = bundle_config['length_x']
                elif key == 'length_y':
                    length_y = bundle_config['length_y']
                elif key == 'length_z':
                    length_z = bundle_config['length_z']
                elif key == 'length_x_abs':
                    length_x_abs = bundle_config['length_x_abs']
                elif key == 'length_y_abs':
                    length_y_abs = bundle_config['length_y_abs']
                elif key == 'length_z_abs':
                    length_z_abs = bundle_config['length_z_abs']
                elif key == 'gt_mask':
                    if gt_dir:
                        gt_mask = os.path.join(gt_dir,
                                               bundle_config['gt_mask'])
                    else:
                        gt_mask = bundle_config['gt_mask']

                    if use_gt_masks_as_all_masks:
                        all_mask = gt_mask
                elif key == 'all_mask':
                    if use_gt_masks_as_all_masks:
                        raise ValueError(
                            "With the option --use_gt_masks_as_all_masks, "
                            "you should not add any all_mask in the config "
                            "file.")
                    if gt_dir:
                        all_mask = os.path.join(gt_dir,
                                                bundle_config['all_mask'])
                    else:
                        all_mask = bundle_config['all_mask']
                elif key == 'endpoints':
                    if 'head' in bundle_config or 'tail' in bundle_config:
                        raise ValueError(
                            "Bundle {} has confusing keywords in the config "
                            "file. Please choose either endpoints OR "
                            "head/tail.".format(bundle))
                    if gt_dir:
                        endpoints = os.path.join(gt_dir,
                                                 bundle_config['endpoints'])
                    else:
                        endpoints = bundle_config['endpoints']
                    roi_option = {'gt_endpoints': endpoints}
                elif key == 'head':
                    if 'tail' not in bundle_config:
                        raise ValueError(
                            "You have provided the head for bundle {}, but "
                            "not the tail".format(bundle))
                    if gt_dir:
                        head = os.path.join(gt_dir, bundle_config['head'])
                        tail = os.path.join(gt_dir, bundle_config['tail'])
                    else:
                        head = bundle_config['head']
                        tail = bundle_config['tail']
                    roi_option = {'gt_head': head, 'gt_tail': tail}
                elif key == 'tail':
                    pass  # dealt with at head
                elif key == 'any_mask':
                    if gt_dir:
                        any_mask = os.path.join(
                            gt_dir, bundle_config['any_mask'])
                    else:
                        any_mask = bundle_config['any_mask']
                else:
                    raise ValueError("Unrecognized value {} in the config "
                                     "file for bundle {}".format(key, bundle))

            angles.append(angle)
            lengths.append(length)
            if length_x is None and length_y is None and length_z is None:
                orientation_lengths.append(None)
            else:
                orientation_lengths.append(
                    [length_x if length_x is not None else def_len,
                     length_y if length_y is not None else def_len,
                     length_z if length_z is not None else def_len])

            if length_x_abs is None and length_y_abs is None and \
                    length_z_abs is None:
                abs_orientation_lengths.append(None)
            else:
                abs_orientation_lengths.append(
                    [length_x_abs if length_x_abs is not None else def_len,
                     length_y_abs if length_y_abs is not None else def_len,
                     length_z_abs if length_z_abs is not None else def_len])
            gt_masks.append(gt_mask)
            all_masks.append(all_mask)
            any_masks.append(any_mask)
            roi_options.append(roi_option)

    if show_warning_gt:
        logging.info(
            "At least one bundle had no gt_mask. Some tractometry metrics "
            "won't be computed (OR, OL) for these bundles.")

    return (bundles, gt_masks, all_masks, any_masks, roi_options,
            lengths, angles, orientation_lengths, abs_orientation_lengths)


def compute_endpoint_masks(roi_options, args):
    """
    If endpoints without heads/tails are loaded, split them and continue
    normally after. Q/C of the output is important. Compatibility between files
    should be already verified.

    Parameters
    ------
    roi_options: dict
        Keys are the bundle names. For each bundle, the value is itself a
        dictionary either key 'gt_endpoints' (the name of the file
        containing the bundle's endpoints), or both keys 'gt_tail' and
        'gt_head' (the names of the respetive files).
    out_dir: str
        Where to save the heads and tails.

    Returns
    -------
    tails, heads: lists of filenames with length the number of bundles.
    """
    tails = []
    heads = []
    for bundle_options in roi_options:
        tail = bundle_options['gt_tail']
        head = bundle_options['gt_head']

        mask_1_img = nib.load(head)
        mask_2_img = nib.load(tail)
        mask_1 = get_data_as_mask(mask_1_img)
        mask_2 = get_data_as_mask(mask_2_img)

        if args.dilate_endpoints:
            mask_1 = binary_dilation(mask_1, iterations=args.dilate_endpoints)
            mask_2 = binary_dilation(mask_2, iterations=args.dilate_endpoints)

        tails.append(mask_2)
        heads.append(mask_1)

    return tails, heads


def compute_masks_from_bundles(gt_files, reference, inverse_mask=False):
    """
    Compute ground-truth masks. If the file is already a mask, load it.
    If it is a bundle, compute the mask. If the filename is None, appends None
    to the lists of masks. Compatibility between files should already be
    verified.

    Parameters
    ----------
    gt_files: list
        List of either StatefulTractograms or niftis.
    parser: ArgumentParser
        Argument parser which handles the script's arguments. Used to print
        parser errors, if any.
    args: Namespace
        List of arguments passed to the script. Used for its 'ref' and
        'bbox_check' arguments.
    inverse_mask: bool
        If true, returns the list of inversed masks instead.

    Returns
    -------
    mask: list[numpy.ndarray]
        The loaded masks.
    """
    save_ref = reference

    gt_bundle_masks = []

    for gt_bundle in gt_files:
        if gt_bundle is not None:
            # Support ground truth as streamlines or masks
            # Will be converted to binary masks immediately
            _, ext = split_name_with_nii(gt_bundle)
            if ext in ['.gz', '.nii.gz']:
                gt_img = nib.load(gt_bundle)
                gt_mask = get_data_as_mask(gt_img)
                dimensions = gt_mask.shape
            else:
                # Cheating ref because it may send a lot of warning if loading
                # many trk with ref (reference was maybe added only for some
                # of these files)
                if ext == '.trk':
                    reference = 'same'
                else:
                    reference = save_ref
                gt_sft = load_tractogram(
                    gt_bundle, reference)
                gt_sft.to_vox()
                gt_sft.to_corner()
                _, dimensions, _, _ = gt_sft.space_attributes
                gt_mask = compute_tract_counts_map(gt_sft.streamlines,
                                                   dimensions).astype(np.int16)
            gt_mask[gt_mask > 0] = 1

            if inverse_mask:
                gt_inv_mask = np.zeros(dimensions, dtype=np.int16)
                gt_inv_mask[gt_mask == 0] = 1
                gt_mask = gt_inv_mask
        else:
            gt_mask = None

        gt_bundle_masks.append(gt_mask)

    return gt_bundle_masks


def _extract_vb_and_wpc_all_bundles(
        gt_tails, gt_heads, sft, bundle_names, lengths, angles,
        orientation_lengths, abs_orientation_lengths, inv_all_masks,
        any_masks, args):
    """
    Loop on every ground truth bundles and extract VS and WPC.

    VS:
       1) Connect the head and tail
       2) Are completely included in the all_mask (if any)
       3) Have acceptable angle, length and length per orientation.
       4) Reach the any_mask (if any)
     +
    WPC connections:
       1) connect the head and tail but criteria 2 and 3 are not respected

    Returns
    -------
    vb_sft_list: list
        List of StatefulTractograms of VS
    wpc_sft_list: list
        List of StatefulTractograms of WPC if args.save_wpc_separately), else
        None.
    all_vs_wpc_ids: list
        List of list of all VS + WPC streamlines detected.
    bundle_stats_dict: dict
        Dictionnary of the processing information for each bundle.

    Saves
    -----
    - Each duplicate in segmented_conflicts/duplicates_*_*.trk
    """
    nb_bundles = len(bundle_names)

    vs_ids_list = []
    wpc_ids_list = []

    remaining_ids = np.arange(len(sft))  # For args.unique management.

    # 1. Extract VB and WPC.
    for i in range(nb_bundles):

        if len(remaining_ids) == 0:
            break

        head_filename = gt_heads[i]
        tail_filename = gt_tails[i]

        vs_ids, wpc_ids = \
            _extract_vb_one_bundle(
                sft[remaining_ids], head_filename, tail_filename, lengths[i],
                angles[i], orientation_lengths[i], abs_orientation_lengths[i],
                inv_all_masks[i], any_masks[i], args.dilate_endpoints)

        if args.unique:
            # Assign actual ids, not from subset
            vs_ids = remaining_ids[vs_ids]
            wpc_ids = remaining_ids[wpc_ids]
            # Update remaining_ids based on valid streamlines only
            remaining_ids = np.setdiff1d(remaining_ids, vs_ids,
                                         assume_unique=True)

        vs_ids_list.append(vs_ids)
        wpc_ids_list.append(wpc_ids)

    all_gt_ids = list(itertools.chain(*vs_ids_list))

    # 2. Remove duplicate WPC and then save.
    if args.save_wpc_separately:
        if args.remove_wpc_belonging_to_another_bundle or args.unique:
            for i in range(nb_bundles):
                new_wpc_ids = np.setdiff1d(wpc_ids_list[i], all_gt_ids)
                wpc_ids_list[i] = new_wpc_ids

        wpc_sft_list = []
        for i in range(nb_bundles):
            wpc_ids = wpc_ids_list[i]
            if len(wpc_ids) == 0:
                wpc_sft = None
            else:
                wpc_sft = sft[wpc_ids]
            wpc_sft_list.append(wpc_sft)
    else:
        # Remove WPCs to be included as IS in the future
        wpc_ids_list = [[] for _ in range(nb_bundles)]
        wpc_sft_list = None

    # 3. If not args.unique, tell users if there were duplicates. Save
    # duplicates separately in segmented_conflicts/duplicates_*_*.trk.
    if not args.unique:
        for i in range(nb_bundles):
            for j in range(i + 1, nb_bundles):
                duplicate_ids = np.intersect1d(vs_ids_list[i], vs_ids_list[j])
                if len(duplicate_ids) > 0:
                    logging.warning(
                        "{} streamlines belong to true connections of both "
                        "bundles {} and {}.\n"
                        "Please verify your criteria!"
                        .format(len(duplicate_ids), bundle_names[i],
                                bundle_names[j]))

    all_vs_ids = np.unique(list(itertools.chain(*vs_ids_list)))
    all_wpc_ids = np.unique(list(itertools.chain(*wpc_ids_list)))
    all_vs_wpc_ids = np.concatenate((all_vs_ids, all_wpc_ids)).astype(int)

    return all_vs_wpc_ids


def _extract_vb_one_bundle(
        sft, head_filename, tail_filename, limits_length, angle,
        orientation_length, abs_orientation_length, inv_all_mask,
        any_mask, dilate_endpoints):
    """
    Extract valid bundle (and valid streamline ids) from a tractogram, based
    on two regions of interest for the endpoints, one region of interest for
    the inclusion of streamlines, and maximum length, maximum angle,
    maximum length per orientation.

    Parameters
    ----------
    sft: StatefulTractogram
        Tractogram containing the streamlines to be extracted.
    head_filename: str
        Filename of the "head" of the bundle.
    tail_filename: str
        Filename of the "tail" of the bundle.
    limits_length: list or None
        Bundle's length parameters: [min max].
    angle: int or None
        Bundle's max angle.
    orientation_length: list or None
        Bundle's length parameters in each direction:
        [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
    abs_orientation_length: idem, computed in absolute values.
    inv_all_mask: np.ndarray or None
        Inverse ALL mask for this bundle: no point must be outside the mask.
    any_mask: np.ndarray or None
        ANY mask for this bundle.
        Streamlines must pass through this mask (touch it) to be included
        in the bundle.
    dilate_endpoints: int or None
        If set, dilate the masks for n iterations.

    Returns
    -------
    vs_ids: list
        List of ids of valid streamlines
    wpc_ids: list
        List of ids of wrong-path connections
    bundle_stats: dict
        Dictionary of recognized streamlines statistics
    """
    mask_1 = head_filename
    mask_2 = tail_filename

    _, vs_ids = filter_grid_roi_both(sft, mask_1, mask_2)

    wpc_ids = []

    # Remove out of inclusion mask (limits_mask)
    if len(vs_ids) > 0 and inv_all_mask is not None:
        tmp_sft = sft[vs_ids]
        _, out_of_mask_ids_from_vs = filter_grid_roi(
            tmp_sft, inv_all_mask, 'any', False)
        out_of_mask_ids = vs_ids[out_of_mask_ids_from_vs]

        # Update ids
        wpc_ids.extend(out_of_mask_ids)
        vs_ids = np.setdiff1d(vs_ids, wpc_ids)

    # Remove streamlines not passing through any_mask
    if len(vs_ids) > 0 and any_mask is not None:
        tmp_sft = sft[vs_ids]
        _, in_mask_ids_from_vs = filter_grid_roi(
            tmp_sft, any_mask, 'any', False)
        in_mask_ids = vs_ids[in_mask_ids_from_vs]

        out_of_mask_ids = np.setdiff1d(vs_ids, in_mask_ids)

        # Update ids
        wpc_ids.extend(out_of_mask_ids)
        vs_ids = in_mask_ids

    return list(vs_ids), list(wpc_ids)


def segment_tractogram_from_roi(
        sft, gt_tails, gt_heads, bundle_names, bundle_lengths, angles,
        orientation_lengths, abs_orientation_lengths, inv_all_masks, any_masks,
        list_rois, args):
    """
    Segments valid bundles (VB). Based on args:
        - args.compute_ic: computes invalid bundles (IB)
        - args.save_wpc_separately: compute WPC

    Returns
    -------
    vb_sft_list: list
        The list of valid bundles discovered. These files are also saved
        in segmented_VB/*_VS.trk.
    wpc_sft_list: list
        The list of wrong path connections: streamlines connecting the right
        endpoint regions but not included in the ALL mask.
        ** This is only computed if args.save_wpc_separately. Else, this is
        None.
    ib_sft_list: list
        The list of invalid bundles: streamlines connecting regions that should
        not be connected.
        ** This is only computed if args.compute_ic. Else, this is None.
    nc_sft_list: list
        The list of rejected streamlines that were not included in any IB.
    ib_names: list
        The list of names for invalid bundles (IB). They are created from the
        combinations of ROIs used for IB computations.
    bundle_stats: dict
        Dictionnary of the processing information for each VB bundle.
    """
    sft.to_vox()

    # VS
    logging.info("Extracting valid bundles (and wpc, if any)")
    detected_vs_wpc_ids = \
        _extract_vb_and_wpc_all_bundles(
            gt_tails, gt_heads, sft, bundle_names, bundle_lengths,
            angles, orientation_lengths, abs_orientation_lengths,
            inv_all_masks, any_masks, args)

    return detected_vs_wpc_ids


class TractometerReward(Reward):

    def __init__(
        self,
        base_dir,
        reference,
        affine_vox2rasmm,
        use_gt_masks_as_all_masks=False
    ):

        self.name = 'tractometer_reward'

        if base_dir is None:
            return

        self.gt_config = os.path.join(base_dir, 'scil_scoring_config.json')

        self.gt_dir = base_dir
        self.reference = reference
        self.affine_vox2rasmm = affine_vox2rasmm

        args_mocker = namedtuple('args', [
            'compute_ic', 'save_wpc_separately', 'unique', 'reference',
            'bbox_check', 'out_dir', 'dilate_endpoints', 'no_empty'])

        temp = tempfile.mkdtemp()
        self.args = args_mocker(
            False, False, True, self.reference, False, temp, 1, False)

        # Load
        (self.gt_tails, self.gt_heads, self.bundle_names, self.list_rois,
         self.bundle_lengths, self.angles, self.orientation_lengths,
         self.abs_orientation_lengths, self.inv_all_masks, self.gt_masks,
         self.any_masks) = \
            load_and_verify_everything(
                reference,
                self.gt_config,
                self.gt_dir,
                use_gt_masks_as_all_masks,
                self.args)

    def __call__(self, streamlines, dones):

        # Change ref of streamlines. This is weird on the ISMRM2015
        # dataset as the diff and anat are not in the same space,
        # but it should be fine on other datasets.
        N = len(streamlines)
        tractogram = Tractogram(
            streamlines=streamlines.copy()[dones])
        tractogram.apply_affine(self.affine_vox2rasmm)
        sft = StatefulTractogram(
            streamlines=tractogram.streamlines,
            reference=self.reference,
            space=Space.RASMM)

        if len(sft.streamlines) == 0:
            return np.zeros((N))

        _, dimensions, _, _ = sft.space_attributes

        # Segment VB, WPC, IB
        detected_vs_wpc_ids = segment_tractogram_from_roi(
            sft, self.gt_tails, self.gt_heads, self.bundle_names,
            self.bundle_lengths, self.angles, self.orientation_lengths,
            self.abs_orientation_lengths, self.inv_all_masks, self.any_masks,
            self.list_rois, self.args)

        reward = np.zeros((N))
        if len(detected_vs_wpc_ids) > 0:
            dones_detected_idx = np.arange((N))[dones][detected_vs_wpc_ids]
            reward[dones_detected_idx] = 1.
        return reward
