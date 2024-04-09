import h5py

from torch.utils.data import Dataset

from TrackToLearn.datasets.utils import SubjectData

device = "cpu"


class SubjectDataset(Dataset):
    """

    """

    def __init__(
        self, file_path: str, dataset_split: str,
    ):
        """
        Args:
        """
        self.file_path = file_path
        self.split = dataset_split
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
        tracking_mask = tracto_data.tracking

        seeding = tracto_data.seeding

        reference = tracto_data.reference

        labels = tracto_data.labels

        connectivity = tracto_data.connectivity

        return (subject_id, input_volume, tracking_mask,
                seeding, peaks, reference, labels, connectivity)

    def __len__(self):
        """
        return the length of the dataset
        """
        return len(self.subjects)
