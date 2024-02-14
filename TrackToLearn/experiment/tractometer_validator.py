import itertools
import json
import logging
import os
import tempfile
from collections import namedtuple

import nibabel as nib
import numpy as np
from dipy.io.streamline import load_tractogram
from scilpy.io.image import get_data_as_mask
from scilpy.segment.tractogram_from_roi import segment_tractogram_from_roi
from scilpy.tractanalysis.scoring import compute_tractometry
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.filenames import split_name_with_nii

from TrackToLearn.experiment.validators import Validator

def_len = [0, np.inf]


def load_and_verify_everything(
    reference,
    gt_config,
    gt_dir,
    use_gt_masks_as_all_masks
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
    gt_tails, gt_heads = compute_endpoint_masks(roi_options)

    # Update the list of every ROI, remove duplicates
    list_rois = gt_tails + gt_heads
    list_rois = list(dict.fromkeys(list_rois))  # Removes duplicates

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


def compute_endpoint_masks(roi_options):
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

        tails.append(tail)
        heads.append(head)

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


class TractometerValidator(Validator):

    def __init__(
        self,
        base_dir,
        reference,
        dilate_endpoints=1,
    ):

        self.name = 'Tractometer'

        self.gt_config = os.path.join(base_dir, 'scil_scoring_config.json')

        self.gt_dir = base_dir
        self.reference = reference
        self.dilation_factor = dilate_endpoints

        # Load
        (self.gt_tails, self.gt_heads, self.bundle_names, self.list_rois,
         self.bundle_lengths, self.angles, self.orientation_lengths,
         self.abs_orientation_lengths, self.inv_all_masks, self.gt_masks,
         self.any_masks) = \
            load_and_verify_everything(
                reference,
                self.gt_config,
                self.gt_dir,
                False)

    def __call__(self, filename, env):

        logging.info("Loading tractogram.")
        sft = load_tractogram(filename, env.reference,
                              bbox_valid_check=True, trk_header_check=True)
        if len(sft.streamlines) == 0:
            return {}

        _, dimensions, _, _ = sft.space_attributes

        args_mocker = namedtuple('args', [
            'compute_ic', 'save_wpc_separately', 'unique', 'reference',
            'bbox_check', 'out_dir', 'dilate_endpoints', 'no_empty'])

        temp = tempfile.mkdtemp()
        args = args_mocker(
            False, False, True, self.reference, False, temp,
            self.dilation_factor, False)

        # Segment VB, WPC, IB
        (vb_sft_list, wpc_sft_list, ib_sft_list, nc_sft,
         ib_names, bundle_stats) = segment_tractogram_from_roi(
            sft, self.gt_tails, self.gt_heads, self.bundle_names,
             self.bundle_lengths, self.angles, self.orientation_lengths,
             self.abs_orientation_lengths, self.inv_all_masks, self.any_masks,
            self.list_rois, args)

        # TODO: return bundle_stats

        # Tractometry on bundles
        final_results = compute_tractometry(
            vb_sft_list, wpc_sft_list, ib_sft_list, nc_sft,
            args, self.bundle_names, self.gt_masks, dimensions, ib_names)

        relevant_results = {'VC': final_results['VS_ratio'],
                            'IC': final_results.get('IC_ratio', 0),
                            'IS': final_results.get('IS_ratio', 0),
                            'NC': final_results.get('NC_ratio', 0),
                            'mean_OL': final_results.get('mean_OL', 0),
                            'VB': final_results['VB'],
                            'IB': final_results.get('IB', 0)}

        return relevant_results
