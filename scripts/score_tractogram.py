#!/usr/bin/env python

from __future__ import division

import argparse
import glob
import logging
import os

from challenge_scoring.io.results import save_results
from challenge_scoring.metrics.scoring import score_submission
from challenge_scoring.utils.attributes import load_attribs
from challenge_scoring.utils.filenames import mkdir


DESCRIPTION = """
    Score a submission for the ISMRM 2015 tractography challenge.

    This is based on the ISMRM 2015 tractography challenge, see
    http://www.tractometer.org/ismrm_2015_challenge/

    This script scores a submission following the method presented in
    https://doi.org/10.1101/084137

    This method differs from the classical Tractometer approach
    (https://doi.org/10.1016/j.media.2013.03.009). Instead of only using
    masks to define the ground truth and classify streamlines in the
    submission, bundles are extracted using a bundle recognition technique.

    More details are provided in the documentation here:
    https://github.com/scilus/ismrm_2015_tractography_challenge_scoring

    The algorithm has 6 main steps:
        1: extract all streamlines that are valid, which are classified as
           Valid Connections (VC) making up Valid Bundles (VB).
        2: remove streamlines shorter than an threshold based on the GT dataset
        3: cluster the remaining streamlines
        4: remove singletons
        5: assign each cluster to the closest ROIs pair. Those make up the
           Invalid Connections (IC), grouped as Invalid Bundles (IB).
        6: streamlines that are neither in VC nor IC are classified as
           No Connection (NC).
"""


def build_args_parser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('tractogram', metavar='TRACTS',
                   help='Tractogram file. File must be tck or trk.')
    p.add_argument('base_dir', metavar='BASE_DIR',
                   help='base directory for scoring data.\n'
                        'See www.tractometer.org/downloads/downloads/'
                        'scoring_data_tractography_challenge.tar.gz')
    p.add_argument('out_dir', metavar='OUT_DIR',
                   help='directory where to send score files')
    p.add_argument('--out_tract_type', choices=['tck', 'trk'], default='tck',
                   help='output type for tracts')
    p.add_argument('--save_full_vc', action='store_true',
                   help='save one file containing all VCs')
    p.add_argument('--save_full_ic', action='store_true',
                   help='save one file containing all ICs')
    p.add_argument('--save_full_nc', action='store_true',
                   help='save one file containing all NCs')
    p.add_argument('--compute_ic_ib', action='store_true',
                   help="Segment rejected streamlines into NC + IC.\n"
                        "Else, all non-vb streamlines are stored as NC.")
    p.add_argument('--save_ib', action='store_true',
                   help='save IB independently.')
    p.add_argument('--save_vb', action='store_true',
                   help='save VB independently.')
    p.add_argument('-f', dest='force', action='store_true',
                   required=False, help='overwrite output files')
    p.add_argument('-v', dest='verbose', action='store_true',
                   required=False, help='produce verbose output')

    return p


def main():
    parser = build_args_parser()
    args = parser.parse_args()

    tractogram = args.tractogram
    base_dir = args.base_dir
    out_dir = args.out_dir

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not os.path.isfile(tractogram):
        parser.error('"{0}" must be a file!'.format(tractogram))

    _, ext = os.path.splitext(tractogram)
    if not (ext == '.tck' or ext == '.trk'):
        parser.error("Tractogram file should be a .tck or .trk, not {}"
                     .format(ext))

    if not os.path.isdir(base_dir):
        parser.error('"{0}" must be a directory!'.format(base_dir))

    scores_dir = mkdir(os.path.join(out_dir, "scores"))
    scores_filename = os.path.join(scores_dir,
                                   os.path.splitext(
                                       os.path.basename(tractogram))[0]
                                   + ".json")

    score_exists = False
    segmented_files = []

    # Check if some results already exist
    if os.path.isfile(scores_filename):
        score_exists = True

    segments_dir = ''
    base_name = ''

    if args.save_full_vc or args.save_full_ic or args.save_ib or args.save_vb \
            or args.save_full_nc:
        segments_dir = mkdir(os.path.join(out_dir, "segmented"))
        base_name = os.path.splitext(os.path.basename(tractogram))[0]

        segmented_files = glob.glob(os.path.join(
            segments_dir, '{}*.{}'.format(base_name, args.out_tract_type)))

    if score_exists or len(segmented_files):
        if not args.force:
            parser.error(
                'Scores file or segmented files already exist.'
                '\nPlease remove or use -f to overwrite.')
        else:
            if score_exists:
                os.remove(scores_filename)
            for f in segmented_files:
                os.remove(f)

    # Basic bundle attributes should be stored in the scoring data directory.
    gt_bundles_attribs_path = os.path.join(args.base_dir,
                                           'gt_bundles_attributes.json')
    if not os.path.isfile(gt_bundles_attribs_path):
        parser.error('Missing the "gt_bundles_attributes.json" file in the '
                     'provided base directory.')

    basic_bundles_attribs = load_attribs(gt_bundles_attribs_path)

    scores = score_submission(tractogram, base_dir, basic_bundles_attribs,
                              args.save_full_vc,
                              args.save_full_ic,
                              args.save_full_nc,
                              args.compute_ic_ib,
                              args.save_ib, args.save_vb,
                              segments_dir, base_name,
                              args.out_tract_type, args.verbose)

    if scores is not None:
        print("Saving results in {}".format(scores_filename))
        save_results(scores_filename, scores)


if __name__ == "__main__":
    main()
