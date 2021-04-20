#!/usr/bin/env python
import argparse

import h5py
import nibabel as nib
import numpy as np

from argparse import RawTextHelpFormatter
from os.path import join as pjoin
from typing import Dict, List, Tuple

from nibabel.nifti1 import Nifti1Image

from TrackToLearn.datasets.processing import normalize_data_volume
from TrackToLearn.utils.utils import (
    Timer)


def parse_args():
    """
    Script to process multiple inputs into a single .hdf5 file.
    Main "signal" (DWI, fODFs, etc.) should be included first.
    Even if it is optional, the WM should be passed as it will be
    used as seeding and tracking mask by Track-to-Learn.
    """

    parser = argparse.ArgumentParser(
        description=parse_args.__doc__,
        formatter_class=RawTextHelpFormatter)
    parser.add_argument('inputs', type=str, nargs='+',
                        help="Inputs to use as signal.")
    parser.add_argument('peaks', type=str,
                        help="Peaks for the reward.")
    parser.add_argument('name', type=str, help="Dataset name")
    parser.add_argument('subject_id', type=str,
                        help="Subject id to use for dataset")
    parser.add_argument('output', type=str,
                        help='Folder to create and output to')
    parser.add_argument('--wm', type=str,
                        help='WM mask.')
    parser.add_argument('--gm', type=str,
                        help='GM mask.')
    parser.add_argument('--csf', type=str,
                        help='CSF mask.')
    parser.add_argument('--interface', type=str,
                        help='Interface mask.')
    parser.add_argument('--normalize', action='store_true',
                        help='If set, normalize input-wise signal.')
    parser.add_argument('--save_signal', action='store_true',
                        help='If set, store the resulting signal volume')
    arguments = parser.parse_args()

    return arguments


def main():
    """ Parse args, generate dataset and save it on disk """
    args = parse_args()

    with Timer("Generating dataset", newline=True):
        generate_dataset(
            input_files=args.inputs,
            peaks_file=args.peaks,
            subject_id=args.subject_id,
            name=args.name,
            output=args.output,
            wm_file=args.wm,
            gm_file=args.gm,
            csf_file=args.csf,
            interface_file=args.interface,
            normalize=args.normalize,
            save_signal=args.save_signal)


def generate_dataset(
    input_files: list,
    peaks_file: str,
    subject_id: str,
    name: str,
    output: str,
    wm_file: str = None,
    gm_file: str = None,
    csf_file: str = None,
    interface_file: str = None,
    normalize: bool = False,
    save_signal: bool = False,
) -> None:
    """ Generate a dataset
    TODO: Docstring
    TODO: Cleanup
    """
    dataset_name = name

    # Clean existing processed files
    dataset_file = pjoin(output, "{}.hdf5".format(dataset_name))

    # Initialize database
    with h5py.File(dataset_file, 'w') as hdf_file:
        # Save version
        hdf_file.attrs['version'] = 2

        with Timer(
            "Processing subject {}".format(subject_id),
            newline=True,
            color='blue'
        ):
            # Process subject's data
            image, peaks, wm_img, gm_img, csf_img, interface_img = \
                process_subject(input_files, peaks_file, subject_id,
                                normalize, wm_file, gm_file, csf_file,
                                interface_file)

            if save_signal:
                nib.save(
                    image,
                    pjoin(output, "{}_signal.nii.gz".format(dataset_name)))

            # Add subject to database
            hdf_subject = hdf_file.create_group(subject_id)

            hdf_input_volume = hdf_subject.create_group('input_volume')
            hdf_input_volume.attrs['vox2rasmm'] = image.affine
            hdf_input_volume.create_dataset('data', data=image.get_fdata())

            hdf_peaks_volume = hdf_subject.create_group('peaks_volume')
            hdf_peaks_volume.attrs['vox2rasmm'] = peaks.affine
            hdf_peaks_volume.create_dataset('data', data=peaks.get_fdata())

            if wm_file:
                hdf_peaks_volume = hdf_subject.create_group('wm_volume')
                hdf_peaks_volume.attrs['vox2rasmm'] = wm_img.affine
                hdf_peaks_volume.create_dataset(
                    'data', data=wm_img.get_fdata())

            if gm_file:
                hdf_peaks_volume = hdf_subject.create_group('gm_volume')
                hdf_peaks_volume.attrs['vox2rasmm'] = gm_img.affine
                hdf_peaks_volume.create_dataset(
                    'data', data=gm_img.get_fdata())

            if csf_file:
                hdf_peaks_volume = hdf_subject.create_group('csf_volume')
                hdf_peaks_volume.attrs['vox2rasmm'] = csf_img.affine
                hdf_peaks_volume.create_dataset(
                    'data', data=csf_img.get_fdata())

            if interface_file:
                hdf_peaks_volume = hdf_subject.create_group('interface_volume')
                hdf_peaks_volume.attrs['vox2rasmm'] = interface_img.affine
                hdf_peaks_volume.create_dataset(
                    'data', data=interface_img.get_fdata())

    print("Saved dataset : {}".format(dataset_file))


def process_subject(
    input_files: str,
    peaks_file: str,
    subject_id: str,
    normalize: bool = False,
    wm: str = None,
    gm: str = None,
    csf: str = None,
    interface: str = None
) -> Tuple[np.ndarray, List, List, Dict, np.ndarray, np.ndarray]:

    affine = nib.load(input_files[0]).affine

    input_volumes = [nib.load(f).get_fdata() for f in input_files]
    for i, v in enumerate(input_volumes):
        if len(v.shape) == 3:
            input_volumes[i] = v[..., None]

    peaks_image = nib.load(peaks_file)

    wm_mask_image = None
    if wm:
        wm_mask_image = nib.load(wm)

    gm_mask_image = None
    if gm:
        gm_mask_image = nib.load(gm)

    csf_mask_image = None
    if csf:
        csf_mask_image = nib.load(csf)

    interface_mask_image = None
    if interface:
        interface_mask_image = nib.load(interface)

    if normalize:
        print('Normalizing signal volume')
        input_volume = normalize_data_volume(input_volumes[0])
    else:
        input_volume = input_volumes[0]

    inputs = np.concatenate([input_volume] + input_volumes[1:], axis=-1)

    # Save processed data
    input_image = Nifti1Image(
        inputs,
        affine)

    return (input_image, peaks_image,
            wm_mask_image, gm_mask_image, csf_mask_image,
            interface_mask_image)


if __name__ == "__main__":
    main()
