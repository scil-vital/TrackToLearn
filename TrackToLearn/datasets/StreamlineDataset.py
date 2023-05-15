import h5py
import numpy as np
import torch

from collections import defaultdict

from dwi_ml.data.processing.space.world_to_vox import convert_world_to_vox
from nibabel.streamlines import Tractogram
from torch.utils.data import Dataset

from TrackToLearn.datasets.utils import SubjectData
from TrackToLearn.environments.utils import (
    get_neighborhood_directions, format_state)
from TrackToLearn.environments.reward import (
    reward_streamlines_step,
)
from TrackToLearn.utils.utils import normalize_vectors

device = "cpu"


class StreamlineDataset(Dataset):
    """
    class that loads hdf5 dataset object
    """

    def __init__(
        self, file_path: str, dataset_split: str, n_dirs=1,
        add_neighborhood=True, dense_rewards=False, reward_scaling=1.0,
        reward_shift=0.0, noise=0.0, device=None
    ):
        """
        Args:
        """
        self.file_path = file_path
        self.split = dataset_split
        self.n_dirs = n_dirs
        self.add_neighborhood = add_neighborhood
        self.dense_rewards = dense_rewards
        self.reward_scaling = reward_scaling
        self.reward_shift = reward_shift
        self.local_reward = False
        self.noise = noise
        with h5py.File(self.file_path, 'r') as f:
            self.normalize = f.attrs['normalize']
            self.step_size = float(f.attrs['step_size'])
            self.subject_list = list(f[dataset_split].keys())
            self.indexes, self.rev_indexes, self.lengths = \
                self._build_indexes(f, dataset_split)
            self.state_size = self._compute_state_size(f)

        # print(self.dense_rewards)

    def _build_indexes(self, dataset_file, split):
        """
        """
        print('Building indexes')
        set_list = list()
        lengths = []
        rev_index = defaultdict(list)

        split_set = dataset_file[split]
        for subject in list(split_set.keys()):
            if subject != 'transitions':
                streamlines = SubjectData.from_hdf_subject(
                    split_set, subject).sft.streamlines
                for i in range(len(streamlines)):
                    k = (subject, i)
                    rev_index[subject].append((len(set_list), i))

                    set_list.append(k)
                lengths.extend(streamlines._lengths)

        print('Done')
        return set_list, rev_index, lengths

    @property
    def archives(self):
        if not hasattr(self, 'f'):
            self.f = h5py.File(self.file_path, 'r')
        return self.f

    def _compute_state_size(self, f):
        subject, strml_idx = self.indexes[0]
        subject_data = SubjectData.from_hdf_subject(f[self.split], subject)
        data_volume = subject_data.input_dv.data

        signal_shape = data_volume.data.shape[-1]

        if self.add_neighborhood:
            signal_shape *= 7

        signal_shape += (3 * self.n_dirs)
        return signal_shape

    def get_one_input(self):

        state_0, *_ = self[0]
        self.f.close()
        del self.f
        return state_0[0]

    def __getitem__(self, index):
        """This method loads, transforms and returns slice corresponding to the
        corresponding index.
        :arg
            index: the index of the slice within patient data
        :return
            A tuple (input, target)
        """
        # return index

        # Map streamline total index -> subject.streamline_id
        subject, strml_idx = self.indexes[index]
        f = self.archives[self.split]
        subject_data = SubjectData.from_hdf_subject(f, subject)
        sft = subject_data.sft.as_sft(strml_idx)
        sft.to_vox()
        streamline = sft.streamlines[0]

        if self.noise > 0.0:
            streamline = streamline + np.random.normal(
                loc=0.0, scale=self.noise, size=streamline.shape)

        data_volume = torch.from_numpy(
            subject_data.input_dv.data)

        # Compute neighborhood positions to add to all streamline chunks, in
        # voxel space.
        input_dv_affine_vox2rasmm = subject_data.input_dv.affine_vox2rasmm
        if self.add_neighborhood:
            step_size_vox = convert_world_to_vox(
                self.step_size,
                input_dv_affine_vox2rasmm)
            neighborhood_directions = torch.tensor(
                get_neighborhood_directions(
                    radius=step_size_vox),
                dtype=torch.float16)

        max_len = len(streamline)
        streamlines = np.tile(streamline[0], (max_len, max_len, 1))
        for i in range(1, max_len + 1):
            streamlines[i-1, -i:, :] = streamline[:i, :]

        # self.render(subject_data.peaks, streamlines)

        signals = torch.from_numpy(format_state(
            streamlines,
            data_volume,
            step_size_vox,
            neighborhood_directions,
            1,
            self.n_dirs,
            device).astype(np.float32))

        states = signals[:-1]
        next_states = signals[1:]

        directions = np.diff(streamline, axis=0)

        actions = torch.from_numpy(directions.astype(np.float32))

        if self.local_reward:
            reward = torch.from_numpy(reward_streamlines_step(
                streamlines,
                subject_data.peaks,
                subject_data.csf,
                subject_data.gm,
                200,
                60,
                2,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                subject_data.input_dv.affine_vox2rasmm
            ).astype(np.float32))

        elif self.dense_rewards:
            reward = torch.from_numpy(
                np.repeat(subject_data.rewards[strml_idx],
                          states.shape[0]).astype(np.float32))
            reward *= self.reward_scaling
        else:
            reward = torch.zeros(states.shape[0], dtype=torch.float32)

        reward[-1] = subject_data.rewards[strml_idx] * self.reward_scaling
        reward += self.reward_shift

        dones = torch.zeros((reward.shape), dtype=torch.float32)
        dones[-1] = 1.

        assert len(states) == len(actions), (len(
            states), len(actions), len(streamline))
        return states, actions, reward, next_states, dones

    def __len__(self):
        """
        return the length of the dataset
        """
        return int(len(self.indexes))

    def render(
        self,
        peaks,
        streamline
    ):
        """ Debug function

        Parameters:
        -----------
        tractogram: Tractogram, optional
            Object containing the streamlines and seeds
        path: str, optional
            If set, save the image at the specified location instead
            of displaying directly
        """
        from fury import window, actor
        # Might be rendering from outside the environment
        tractogram = Tractogram(
            streamlines=streamline,
            data_per_streamline={
                'seeds': streamline[:, 0, :]
            })

        # Reshape peaks for displaying
        X, Y, Z, M = peaks.data.shape
        peaks = np.reshape(peaks.data, (X, Y, Z, 5, M//5))

        # Setup scene and actors
        scene = window.Scene()

        stream_actor = actor.streamtube(tractogram.streamlines)
        peak_actor = actor.peak_slicer(peaks,
                                       np.ones((X, Y, Z, M)),
                                       colors=(0.2, 0.2, 1.),
                                       opacity=0.5)
        dot_actor = actor.dots(tractogram.data_per_streamline['seeds'],
                               color=(1, 1, 1),
                               opacity=1,
                               dot_size=2.5)
        scene.add(stream_actor)
        scene.add(peak_actor)
        scene.add(dot_actor)
        scene.reset_camera_tight(0.95)

        showm = window.ShowManager(scene, reset_camera=True)
        showm.initialize()
        showm.start()
