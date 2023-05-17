#!/usr/bin/env python
import argparse

import h5py
import json
import nibabel as nib
import numpy as np

from argparse import RawTextHelpFormatter
from os.path import join

from nibabel.nifti1 import Nifti1Image
from scilpy.io.utils import add_sh_basis_args

from TrackToLearn.datasets.processing import min_max_normalize_data_volume
from TrackToLearn.utils.utils import (
    Timer)

"""
Script to process "multiple" subjects into a single .hdf5 file.
See example configuration file.

Heavly inspired by https://github.com/scil-vital/dwi_ml/blob/master/dwi_ml/data/hdf5/hdf5_creation.py # noqa E405
But modified to suit my needs.
"""


def parse_args():

    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)
    parser.add_argument('path', type=str,
                        help='Location of the dataset files.')
    parser.add_argument('config_file', type=str,
                        help="Configuration file to load subjects and their"
                        " volumes.")
    parser.add_argument('output', type=str,
                        help="Output filename including path")
    parser.add_argument('--normalize', action='store_true',
                        help='If set, normalize first input signal.')

    basis_group = parser.add_argument_group('Basis options')
    add_sh_basis_args(basis_group)

    arguments = parser.parse_args()
    if arguments.sh_basis == 'tournier07':
        parser.error('Only descoteaux07 basis is supported')
    return arguments


def main():
    """ Parse args, generate dataset and save it on disk """
    args = parse_args()

    with Timer("Generating dataset", newline=True):
        generate_dataset(path=args.path,
                         config_file=args.config_file,
                         output=args.output,
                         normalize=args.normalize)


def generate_dataset(
    path: str,
    config_file: str,
    output: str,
    normalize: bool = False,
) -> None:
    """ Generate a dataset

    Args:
        config_file:
        output:
        normalize:

    """

    dataset_name = output

    # Clean existing processed files
    dataset_file = "{}.hdf5".format(dataset_name)

    # Initialize database
    with h5py.File(dataset_file, 'w') as hdf_file:
        # Save version
        hdf_file.attrs['version'] = 2
        hdf_file.attrs['normalize'] = normalize is True

        with open(join(path, config_file), "r") as conf:
            config = json.load(conf)

            add_subjects_to_hdf5(
                path, config, hdf_file, "training", normalize)

            add_subjects_to_hdf5(
                path, config, hdf_file, "validation", normalize)

            add_subjects_to_hdf5(
                path, config, hdf_file, "testing", normalize)

    print("Saved dataset : {}".format(dataset_file))


def add_subjects_to_hdf5(
    path, config, hdf_file, dataset_split, normalize,
):
    """

    Args:
        config:
        hdf_file:
        dataset_split:
        normalize:

    """

    hdf_split = hdf_file.create_group(dataset_split)
    for subject_id in config[dataset_split]:
        with Timer(
            "Processing subject {}".format(subject_id),
            newline=True,
            color='blue'
        ):

            subject_config = config[dataset_split][subject_id]
            hdf_subject = hdf_split.create_group(subject_id)
            add_subject_to_hdf5(path, subject_config, hdf_subject, normalize)


def add_subject_to_hdf5(
    path, config, hdf_subject, normalize,
):
    """

    Args:
        config:
        hdf_subject:
        normalize:

    """

    input_files = config['inputs']
    peaks_file = config['peaks']
    wm_file = config['wm']
    gm_file = config['gm']
    csf_file = config['csf']
    interface_file = config['interface']
    include_file = config['include']
    exclude_file = config['exclude']

    # Process subject's data
    process_subject(hdf_subject, path, input_files, peaks_file, wm_file,
                    gm_file, csf_file, interface_file, include_file,
                    exclude_file, normalize)


def process_subject(
    hdf_subject,
    path: str,
    inputs: str,
    peaks: str,
    wm: str,
    gm: str,
    csf: str,
    interface: str,
    include: str,
    exclude: str,
    normalize: bool,
):
    """

    Args:
        hdf_subject:
        inputs:
        peaks:
        wm:
        gm:
        csf:
        interface:
        include:
        exclude:
        normalize:

    """

    ref_volume = nib.load(join(path, inputs[0]))
    affine = ref_volume.affine
    header = ref_volume.header

    input_volumes = [nib.load(join(path, f)).get_fdata() for f in inputs]
    print('Using as inputs', inputs)
    for i, v in enumerate(input_volumes):
        if len(v.shape) == 3:
            input_volumes[i] = v[..., None]

    if normalize:
        print('Normalizing first signal volume')
        input_volume = min_max_normalize_data_volume(input_volumes[0])
    else:
        input_volume = input_volumes[0]

    signal = np.concatenate([input_volume] + input_volumes[1:], axis=-1)
    # Save processed data
    signal_image = Nifti1Image(
        signal,
        affine,
        header)

    add_volume_to_hdf5(hdf_subject, signal_image, 'input_volume')

    peaks_image = nib.load(join(path, peaks))
    add_volume_to_hdf5(hdf_subject, peaks_image, 'peaks_volume')

    wm_mask_image = nib.load(join(path, wm))
    add_volume_to_hdf5(hdf_subject, wm_mask_image, 'wm_volume')

    gm_mask_image = nib.load(join(path, gm))
    add_volume_to_hdf5(hdf_subject, gm_mask_image, 'gm_volume')

    csf_mask_image = nib.load(join(path, csf))
    add_volume_to_hdf5(hdf_subject, csf_mask_image, 'csf_volume')

    interface_mask_image = nib.load(join(path, interface))
    add_volume_to_hdf5(hdf_subject, interface_mask_image, 'interface_volume')

    include_mask_image = nib.load(join(path, include))
    add_volume_to_hdf5(hdf_subject, include_mask_image, 'include_volume')

    exclude_mask_image = nib.load(join(path, exclude))
    add_volume_to_hdf5(hdf_subject, exclude_mask_image, 'exclude_volume')


def add_volume_to_hdf5(hdf_subject, volume_img, volume_name):
    """

    Args:
        hdf_subject:
        volume_img:
        volume_name:

    """

    hdf_input_volume = hdf_subject.create_group(volume_name)
    hdf_input_volume.attrs['vox2rasmm'] = volume_img.affine
    hdf_input_volume.create_dataset('data', data=volume_img.get_fdata())


if __name__ == "__main__":
    main()
