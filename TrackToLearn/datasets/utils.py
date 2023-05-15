import numpy as np

from dwi_ml.data.dataset.streamline_containers import LazySFTData


class MRIDataVolume(object):
    """
    Class used to encapsulate MRI metadata alongside a data volume,
    such as the vox2rasmm affine or the subject_id.
    """

    def __init__(
        self, data=None, affine_vox2rasmm=None, subject_id=None, filename=None
    ):
        self._data = data
        self.affine_vox2rasmm = affine_vox2rasmm
        self.subject_id = subject_id
        self.filename = filename

    @classmethod
    def from_hdf_group(cls, hdf, group, default=None):
        """ Create an MRIDataVolume from an HDF group object """
        try:
            data = np.array(hdf[group]['data'], dtype=np.float32)
            affine_vox2rasmm = np.array(
                hdf[group].attrs['vox2rasmm'], dtype=np.float32)
        except KeyError:
            print(
                "{} is absent from {}, replacing it with empty volume.".format(
                    group, hdf))
            data = np.zeros_like(hdf[default]['data'], dtype=np.float32)
            affine_vox2rasmm = np.array(
                hdf[default].attrs['vox2rasmm'], dtype=np.float32)
        return cls(data=data, affine_vox2rasmm=affine_vox2rasmm)

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        """ Return the shape of the data """
        return self.data.shape


class SubjectData(object):
    """
    Tractography-related data (input information, tracking mask, peaks and
    streamlines)
    """

    def __init__(
        self,
        subject_id: str,
        input_dv=None,
        peaks=None,
        wm=None,
        gm=None,
        csf=None,
        include=None,
        exclude=None,
        interface=None,
        sft=None,
        rewards=None,
        states=None
    ):
        self.subject_id = subject_id
        self.input_dv = input_dv
        self.peaks = peaks
        self.wm = wm
        self.gm = gm
        self.csf = csf
        self.include = include
        self.exclude = exclude
        self.interface = interface
        self.rewards = rewards
        self.states = states
        self.sft = sft

    @classmethod
    def from_hdf_subject(cls, hdf_file, subject_id):
        """ Create a SubjectData object from an HDF group object """
        hdf_subject = hdf_file[subject_id]
        input_dv = MRIDataVolume.from_hdf_group(hdf_subject, 'input_volume')

        peaks = MRIDataVolume.from_hdf_group(hdf_subject, 'peaks_volume')
        wm = MRIDataVolume.from_hdf_group(hdf_subject, 'wm_volume')
        gm = MRIDataVolume.from_hdf_group(hdf_subject, 'gm_volume')
        csf = MRIDataVolume.from_hdf_group(
            hdf_subject, 'csf_volume', 'wm_volume')
        include = MRIDataVolume.from_hdf_group(
            hdf_subject, 'include_volume', 'wm_volume')
        exclude = MRIDataVolume.from_hdf_group(
            hdf_subject, 'exclude_volume', 'wm_volume')
        interface = MRIDataVolume.from_hdf_group(
            hdf_subject, 'interface_volume', 'wm_volume')

        states = None
        sft = None
        rewards = None
        if 'streamlines' in hdf_subject:
            sft = LazySFTData.init_from_hdf_info(
                hdf_subject['streamlines'])
            rewards = np.array(hdf_subject['streamlines']['rewards'])

        return cls(
            subject_id, input_dv=input_dv, wm=wm, gm=gm, csf=csf,
            include=include, exclude=exclude, interface=interface,
            peaks=peaks, sft=sft, rewards=rewards, states=states)


def convert_length_mm2vox(
    length_mm: float, affine_vox2rasmm: np.ndarray
) -> float:
    """Convert a length from mm to voxel space (if the space is isometric).

    Parameters
    ----------
    length_mm : float
        Length in mm.
    affine_vox2rasmm : np.ndarray
        Affine to bring coordinates from voxel space to rasmm space, usually
        provided with an anatomy file.

    Returns
    -------
    length_vox : float
        Length expressed in isometric voxel units.

    Raises
    ------
    ValueError
        If the voxel space is not isometric
        (different absolute values on the affine diagonal).
    """
    diag = np.diagonal(affine_vox2rasmm)[:3]
    vox2mm = np.mean(np.abs(diag))

    # Affine diagonal should have the same absolute value
    # for an isometric space
    if not np.allclose(np.abs(diag), vox2mm, rtol=5e-2, atol=5e-2):
        raise ValueError("Voxel space is not iso, "
                         " cannot convert a scalar length "
                         "in mm to voxel space. "
                         "Affine provided : {}".format(affine_vox2rasmm))

    length_vox = length_mm / vox2mm
    return length_vox
