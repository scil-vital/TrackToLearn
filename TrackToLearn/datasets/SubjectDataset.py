import h5py

from torch.utils.data import Dataset

from TrackToLearn.datasets.utils import SubjectData

device = "cpu"


class SubjectDataset(Dataset):
    """

    """

    def __init__(
        self, file_path: str, dataset_split: str, interface_seeding: bool,
    ):
        """
        Args:
        """
        self.file_path = file_path
        self.split = dataset_split
        self.interface_seeding = interface_seeding
        with h5py.File(self.file_path, 'r') as f:
            self.subjects = list(f[dataset_split].keys())

    @property
    def archives(self):
        if not hasattr(self, 'f'):
            self.f = h5py.File(self.file_path, 'r')[self.split]
        return self.f

    def __getitem__(self, index):
        """
        """

        # return index
        subject_id = self.subjects[index]

        tracto_data = SubjectData.from_hdf_subject(
            self.archives, subject_id)

        tracto_data.input_dv.subject_id = subject_id
        input_volume = tracto_data.input_dv

        # Load peaks for reward
        peaks = tracto_data.peaks

        # Load tracking mask
        tracking_mask = tracto_data.wm

        # Load target and exclude masks
        target_mask = tracto_data.gm

        include_mask = tracto_data.include
        exclude_mask = tracto_data.exclude

        if self.interface_seeding:
            seeding = tracto_data.interface
        else:
            seeding = tracto_data.wm

        reference = tracto_data.reference

        return (subject_id, input_volume, tracking_mask, include_mask,
                exclude_mask, target_mask, seeding, peaks, reference)

    def __len__(self):
        """
        return the length of the dataset
        """
        return len(self.subjects)
