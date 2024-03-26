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

from TrackToLearn.utils.utils import (
    Timer)

"""
Script to process "multiple" subjects into a single .hdf5 file.
See example configuration file.

Heavly inspired by https://github.com/scil-vital/dwi_ml/blob/master/dwi_ml/data/hdf5/hdf5_creation.py # noqa E405
But modified to suit my needs.
"""


def generate_dataset(
    path: str,
    config_file: str,
    output: str,
) -> None:
    """ Generate a dataset

    Args:
        config_file:
        output:

    """

    dataset_name = output

    # Clean existing processed files
    dataset_file = "{}.hdf5".format(dataset_name)

    # Initialize database
    with h5py.File(dataset_file, 'w') as hdf_file:
        # Save version
        hdf_file.attrs['version'] = 2

        with open(join(path, config_file), "r") as conf:
            config = json.load(conf)
            print(config.keys())
            add_subjects_to_hdf5(
                path, config, hdf_file, "training")

            add_subjects_to_hdf5(
                path, config, hdf_file, "validation")

            add_subjects_to_hdf5(
                path, config, hdf_file, "testing")

    print("Saved dataset : {}".format(dataset_file))


def add_subjects_to_hdf5(
    path, config, hdf_file, dataset_split,
):
    """

    Args:
        config:
        hdf_file:
        dataset_split:

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
            add_subject_to_hdf5(path, subject_config, hdf_subject)


def add_subject_to_hdf5(
    path, config, hdf_subject,
):
    """

    Args:
        config:
        hdf_subject:

    """

    input_files = config['inputs']
    peaks_file = config['peaks']
    tracking_file = config['tracking']
    seeding_file = config['seeding']
    anat_file = config['anat']

    # Process subject's data
    process_subject(hdf_subject, path, input_files, peaks_file, tracking_file,
                    seeding_file, anat_file)


def process_subject(
    hdf_subject,
    path: str,
    inputs: str,
    peaks: str,
    tracking: str,
    seeding: str,
    anat: str,
):
    """ Process a subject's data and save it in the hdf5 file.

    Parameters
    ----------
    hdf_subject : h5py.Group
        HDF5 group to save the data.
    path : str
        Path to the data.
    inputs : list of str
        List of input files.
    peaks : str
        Peaks file.
    tracking : str
        Tracking mask file.
    seeding : str
        Seeding mask file.
    anat : str
        Anatomical file.
    """

    ref_volume = nib.load(join(path, inputs[0]))
    affine = ref_volume.affine
    header = ref_volume.header

    input_volumes = [nib.load(join(path, f)).get_fdata() for f in inputs]
    print('Using as inputs', inputs)
    for i, v in enumerate(input_volumes):
        if len(v.shape) == 3:
            input_volumes[i] = v[..., None]
    input_volume = input_volumes[0]

    signal = np.concatenate([input_volume] + input_volumes[1:], axis=-1)

    signal_image = Nifti1Image(
        signal,
        affine,
        header)

    add_volume_to_hdf5(hdf_subject, signal_image, 'input_volume')

    peaks_image = nib.load(join(path, peaks))
    add_volume_to_hdf5(hdf_subject, peaks_image, 'peaks_volume')

    tracking_mask_image = nib.load(join(path, tracking))
    add_volume_to_hdf5(hdf_subject, tracking_mask_image, 'tracking_volume')

    seeding_mask_image = nib.load(join(path, seeding))
    add_volume_to_hdf5(hdf_subject, seeding_mask_image, 'seeding_volume')

    anat_image = nib.load(join(path, anat))
    add_volume_to_hdf5(hdf_subject, anat_image, 'anat_volume')


def add_volume_to_hdf5(hdf_subject, volume_img, volume_name):
    """ Add a volume to the hdf5 file.

    Parameters
    ----------
    hdf_subject : h5py.Group
        HDF5 group to save the data.
    volume_img : nibabel.Nifti1Image
        Volume to save.
    volume_name : str
        Name of the volume.
    """

    hdf_input_volume = hdf_subject.create_group(volume_name)
    hdf_input_volume.attrs['vox2rasmm'] = volume_img.affine
    hdf_input_volume.create_dataset('data', data=volume_img.get_fdata())


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
                         output=args.output)


if __name__ == "__main__":
    main()
