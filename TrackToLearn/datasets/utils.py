import numpy as np


class MRIDataVolume(object):
    """
    Class used to encapsulate MRI metadata alongside a data volume,
    such as the vox2rasmm affine or the subject_id.
    """

    def __init__(self, data=None, affine_vox2rasmm=None, subject_id=None):
        self._data = data
        self.affine_vox2rasmm = affine_vox2rasmm
        self.subject_id = subject_id

    @classmethod
    def from_hdf_group(cls, hdf_group):
        """ Create an MRIDataVolume from an HDF group object """
        data = np.array(hdf_group['data'], dtype=np.float32)
        affine_vox2rasmm = np.array(
            hdf_group.attrs['vox2rasmm'], dtype=np.float32)
        return cls(data=data, affine_vox2rasmm=affine_vox2rasmm)

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        """ Return the shape of the data """
        return self.data.shape


class TractographyData(object):
    """
    Tractography-related data (input information, tracking mask, peaks)
    """

    def __init__(
      self,
      input_dv=None,
      tracking=None,
      exclude=None,
      target=None,
      seeding=None,
      peaks=None,
      lengths_mm=None,
    ):
        self.input_dv = input_dv
        self._tracking = tracking
        self._exclude = exclude
        self._target = target
        self._seeding = seeding
        self._peaks = peaks
        self.lengths_mm = lengths_mm

    @property
    def peaks(self):
        return self._peaks

    @property
    def tracking(self):
        return self._tracking

    @property
    def exclude(self):
        return self._exclude

    @property
    def target(self):
        return self._target

    @property
    def seeding(self):
        return self._seeding

    @classmethod
    def from_hdf_subject(cls, hdf_subject):
        """ Create a TractographyData object from an HDF group object """
        input_dv = MRIDataVolume.from_hdf_group(hdf_subject['input_volume'])
        peaks = MRIDataVolume.from_hdf_group(hdf_subject['peaks_volume'])
        tracking = MRIDataVolume.from_hdf_group(hdf_subject['wm_volume'])
        exclude = MRIDataVolume.from_hdf_group(hdf_subject['csf_volume'])
        target = MRIDataVolume.from_hdf_group(hdf_subject['gm_volume'])

        seeding = None
        if 'seeding_volume' in hdf_subject:
            seeding = MRIDataVolume.from_hdf_group(
                hdf_subject['seeding_volume'])

        lengths_mm = None

        return cls(
            input_dv=input_dv, tracking=tracking, exclude=exclude,
            target=target, seeding=seeding, peaks=peaks, lengths_mm=lengths_mm)


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
