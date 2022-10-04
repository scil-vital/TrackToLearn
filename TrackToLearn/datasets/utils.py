import random
import numpy as np
import torch

from functools import partial

from dwi_ml.data.dataset.streamline_containers import LazySFTData
from torch.nn.utils.rnn import pack_sequence


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


class BucketSampler(torch.utils.data.Sampler):
    """ Construct batches to fit as many sequences possible
    given the batch size.
    """

    def __init__(
        self, lengths, shuffle=True,
        batch_size=32, drop_last=False,
        number_of_batches=None,
    ):

        super().__init__(lengths)

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.lengths = lengths
        self.number_of_batches = number_of_batches
        self.__iter__()

    def __iter__(self):
        """ Return an iterator filled with streamline indexes.
        First iteration is slow, otherwise the rest is fast.
        """

        indices = list(range(len(self.lengths)))

        # Shuffle whole episodes dataset
        if self.shuffle is True:
            random.shuffle(indices)

        # Batch streamlines so that total number of points <= batch size
        batches = []
        curr_batch = []
        curr_batch_length = 0
        for i in indices:
            le = self.lengths[i]
            # If the episode fits in the batch, add it
            if le + curr_batch_length < self.batch_size:
                curr_batch.append(i)
                curr_batch_length += le
            # Else the batch if full, add it to the iterator
            else:
                batches.append(curr_batch)
                curr_batch = [i]
                curr_batch_length = le

            # Constraint the number of batches to train for N * M gradient
            # steps. e.g. 500 epochs * 250 gradient steps per epoch = 125k
            # gradient steps.
            if (self.number_of_batches and
                    len(batches) == self.number_of_batches):
                break

        # Length will be used to compute the epoch length
        self.length = len(batches)

        return iter(batches)

    def __len__(self):
        return self.length


class Collater:
    def __init__(self, dataset, is_recurrent):
        self.dataset = dataset
        self.is_recurrent = is_recurrent
        if is_recurrent:
            self.collate_fn = self.collate_sequences
        else:
            self.collate_fn = self.collate_batch

    def collate_batch(self, data):

        states, targets, rewards, next_states, dones = \
            map(torch.cat, zip(*[i for i in data]))

        return (states, targets,
                rewards, next_states, dones)

    def collate_sequences(self, data):
        pack_fn = partial(pack_sequence, enforce_sorted=False)
        states, targets, rewards, next_states, dones = \
            map(pack_fn, zip(*[i for i in data]))
        return (states, targets,
                rewards, next_states, dones)

    def __call__(self, data):
        return self.collate_fn(data)


class SupervisedCollater:
    def __init__(self, dataset, is_recurrent):
        self.dataset = dataset
        self.is_recurrent = is_recurrent
        if is_recurrent:
            self.collate_fn = self.collate_sequences
        else:
            self.collate_fn = self.collate_batch

    def collate_batch(self, data):
        states, targets = \
            map(torch.cat, zip(*[i for i in data]))
        indices = list(range(states.shape[0]))

        random.shuffle(indices)
        return states[indices], targets[indices]

    def collate_sequences(self, data):
        pack_fn = partial(pack_sequence, enforce_sorted=False)
        states, targets = \
            map(pack_fn, zip(*[i for i in data]))
        return states, targets

    def __call__(self, data):
        return self.collate_fn(data)


class TransitionData(object):
    """
    """

    def __init__(
        self,
        subject_id: str,
        states=None,
        next_states=None,
        actions=None,
        rewards=None,
        dones=None,
    ):
        self.subject_id = subject_id
        self.states = states
        self.next_states = next_states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones

    @classmethod
    def from_hdf_subject(cls, hdf_file, subject_id, i):
        """ Create a SubjectData object from an HDF group object """
        hdf_transitions = hdf_file['transitions']

        states = np.array(
            hdf_transitions['states']['data'][i], dtype=np.float32)
        next_states = np.array(
            hdf_transitions['next_states']['data'][i], dtype=np.float32)
        actions = np.array(
            hdf_transitions['actions']['data'][i], dtype=np.float32)
        rewards = np.array(
            hdf_transitions['rewards']['data'][i], dtype=np.float32)
        dones = np.array(hdf_transitions['dones']['data'][i], dtype=np.float32)

        return states, next_states, actions, rewards, dones
