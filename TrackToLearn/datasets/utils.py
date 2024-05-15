import nibabel as nib
import numpy as np

from dipy.data import get_sphere
from dipy.reconst.csdeconv import sph_harm_ind_list
from scilpy.reconst.utils import get_sh_order_and_fullness
from scilpy.reconst.sh import convert_sh_basis


class MRIDataVolume(object):
    """
    Class used to encapsulate MRI metadata alongside a data volume,
    such as the vox2rasmm affine or the subject_id.
    """

    def __init__(
        self, data=None, affine_vox2rasmm=None
    ):
        self._data = data
        self.affine_vox2rasmm = affine_vox2rasmm

    @classmethod
    def from_hdf_group(cls, hdf, group, default=None):
        """ Create an MRIDataVolume from an HDF group object """
        try:
            data = np.array(hdf[group]['data'], dtype=np.float32)
            affine_vox2rasmm = np.array(
                hdf[group].attrs['vox2rasmm'], dtype=np.float32)
        except KeyError:
            print('Missing {} from dataset'.format(group))
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
        tracking=None,
        seeding=None,
        reference=None,
    ):
        self.subject_id = subject_id
        self.input_dv = input_dv
        self.peaks = peaks
        self.tracking = tracking
        self.seeding = seeding
        self.reference = reference

    @classmethod
    def from_hdf_subject(cls, hdf_file, subject_id):
        """ Create a SubjectData object from an HDF group object """
        hdf_subject = hdf_file[subject_id]
        input_dv = MRIDataVolume.from_hdf_group(hdf_subject, 'input_volume')

        peaks = MRIDataVolume.from_hdf_group(hdf_subject, 'peaks_volume')
        tracking = MRIDataVolume.from_hdf_group(hdf_subject, 'tracking_volume')
        seeding = MRIDataVolume.from_hdf_group(
            hdf_subject, 'seeding_volume', 'tracking_volume')
        anatomy = MRIDataVolume.from_hdf_group(
            hdf_subject, 'anat_volume', 'tracking_volume')

        reference = nib.Nifti1Image(anatomy.data, anatomy.affine_vox2rasmm)

        return cls(
            subject_id, input_dv=input_dv, tracking=tracking,
            seeding=seeding, reference=reference, peaks=peaks)


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


def set_sh_order_basis(
    sh,
    sh_basis,
    target_basis='descoteaux07',
    target_order=6,
    sphere_name='repulsion724',
):
    """ Convert SH to the target basis and order. In practice, it is always
    order 6 and descoteaux07 basis.

    This uses a lot of "hacks" to convert the ODFs. To go from full to
    symmetric basis, only even coefficents are selected.

    To go from order N to order 6, SH coefficients are either truncated
    or padded.

    """

    sphere = get_sphere(sphere_name)

    n_coefs = sh.shape[-1]
    sh_order, full_basis = get_sh_order_and_fullness(n_coefs)
    sh_order = int(sh_order)

    # If SH in full basis, convert them
    if full_basis is True:
        print('SH coefficients are in "full" basis, only even coefficients '
              'will be used.')
        _, orders = sph_harm_ind_list(sh_order, full_basis)
        sh = sh[..., orders % 2 == 0]

    # If SH are not of target order, convert them
    if sh_order != target_order:
        print('SH coefficients are of order {}, '
              'converting them to order {}.'.format(sh_order, target_order))
        target_n_coefs = len(sph_harm_ind_list(target_order)[0])

        if n_coefs > target_n_coefs:
            sh = sh[..., :target_n_coefs]
        else:
            X, Y, Z = sh.shape[:3]
            n_missing_coefs = target_n_coefs - n_coefs
            sh = np.concatenate(
                (sh, np.zeros((X, Y, Z, n_missing_coefs))), axis=-1)

    # If SH are not in the descoteaux07 basis, convert them
    if sh_basis != target_basis:
        print('SH coefficients are in the {} basis, '
              'converting them to {}.'.format(sh_basis, target_basis))
        sh = convert_sh_basis(
            sh, sphere, input_basis=sh_basis, nbr_processes=1)

    return sh
