import logging
import os

import numpy as np
import lmdb as lmdb
import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from data_loader.data_preprocessor import DataPreprocessor
import pyarrow


def word_seq_collate_fn(data):
    """ collate function for loading word sequences in variable lengths """
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # separate source and target sequences
    word_seq, poses_seq, audio, aux_info = zip(*data)

    # merge sequences
    words_lengths = torch.LongTensor([len(x) for x in word_seq])
    word_seq = pad_sequence(word_seq, batch_first=True).long()

    poses_seq = default_collate(poses_seq)
    audio = default_collate(audio)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return word_seq, words_lengths, poses_seq, audio, aux_info


class TwhDataset(Dataset):
    def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, data_mean, data_std):

        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.lang_model = None
        self.data_mean = np.array(data_mean).squeeze()
        self.data_std = np.array(data_std).squeeze()

        logging.info("Reading data '{}'...".format(lmdb_dir))
        preloaded_dir = lmdb_dir + '_cache'
        if not os.path.exists(preloaded_dir):
            data_sampler = DataPreprocessor(lmdb_dir, preloaded_dir, n_poses,
                                            subdivision_stride, pose_resampling_fps)
            data_sampler.run()
        else:
            logging.info('Found pre-loaded samples from {}'.format(preloaded_dir))

        # init lmdb
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)
            word_seq, pose_seq, audio, aux_info = sample

        def words_to_tensor(lang, words, end_time=None):
            indexes = [lang.SOS_token]
            for word in words:
                if end_time is not None and word[1] > end_time:
                    break
                indexes.append(lang.get_word_index(word[0]))
            indexes.append(lang.EOS_token)
            return torch.Tensor(indexes).long()

        # normalize
        std = np.clip(self.data_std, a_min=0.01, a_max=None)
        pose_seq = (pose_seq - self.data_mean) / std

        # to tensors
        if self.lang_model:
            word_seq_tensor = words_to_tensor(self.lang_model, word_seq, aux_info['end_time'])
        else:
            word_seq_tensor = 0
        pose_seq = torch.from_numpy(pose_seq).reshape((pose_seq.shape[0], -1)).float()
        audio = torch.from_numpy(audio).float()

        return word_seq_tensor, pose_seq, audio, aux_info

    def set_lang_model(self, lang_model):
        self.lang_model = lang_model
