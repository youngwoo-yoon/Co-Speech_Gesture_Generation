""" create data samples """
import lmdb
import math
import numpy as np
import pyarrow


class DataPreprocessor:
    def __init__(self, clip_lmdb_dir, out_lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps):
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps

        self.src_lmdb_env = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
        with self.src_lmdb_env.begin() as txn:
            self.n_videos = txn.stat()['entries']

        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * 16000)

        # create db for samples
        map_size = 1024 * 50  # in MB
        map_size <<= 20  # in B
        self.dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        self.n_out_samples = 0

    def run(self):
        src_txn = self.src_lmdb_env.begin(write=False)

        # sampling and normalization
        cursor = src_txn.cursor()
        for key, value in cursor:
            video = pyarrow.deserialize(value)
            vid = video['vid']
            clips = video['clips']
            for clip_idx, clip in enumerate(clips):
                self._sample_from_clip(vid, clip)

        # print stats
        with self.dst_lmdb_env.begin() as txn:
            print('no. of samples: ', txn.stat()['entries'])

        # close db
        self.src_lmdb_env.close()
        self.dst_lmdb_env.sync()
        self.dst_lmdb_env.close()

    def _sample_from_clip(self, vid, clip):
        clip_skeleton = clip['poses']
        clip_audio_raw = clip['audio_raw']
        clip_word_list = clip['words']

        # divide
        aux_info = []
        sample_skeletons_list = []
        sample_words_list = []
        sample_audio_list = []
        
        num_subdivision = math.floor(
            (len(clip_skeleton) - self.n_poses)
            / self.subdivision_stride) + 1  # floor((K - (N+M)) / S) + 1

        for i in range(num_subdivision):
            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses

            sample_skeletons = clip_skeleton[start_idx:fin_idx]
            subdivision_start_time = start_idx / self.skeleton_resampling_fps
            subdivision_end_time = fin_idx / self.skeleton_resampling_fps
            sample_words = self.get_words_in_time_range(word_list=clip_word_list,
                                                        start_time=subdivision_start_time,
                                                        end_time=subdivision_end_time)

            # filtering
            if len(sample_words) < 3:
                continue

            # raw audio
            audio_start = math.floor(start_idx / len(clip_skeleton) * len(clip_audio_raw))
            audio_end = audio_start + self.audio_sample_length
            sample_audio = clip_audio_raw[audio_start:audio_end]

            motion_info = {'vid': vid,
                           'start_frame_no': start_idx,
                           'end_frame_no': fin_idx,
                           'start_time': subdivision_start_time,
                           'end_time': subdivision_end_time}

            sample_skeletons_list.append(sample_skeletons)
            sample_words_list.append(sample_words)
            sample_audio_list.append(sample_audio)
            aux_info.append(motion_info)

        if len(sample_skeletons_list) > 0:
            with self.dst_lmdb_env.begin(write=True) as txn:
                for words, poses, audio, aux in zip(sample_words_list, sample_skeletons_list,
                                                    sample_audio_list, aux_info):
                    poses = np.asarray(poses)

                    # save
                    k = '{:010}'.format(self.n_out_samples).encode('ascii')
                    v = [words, poses, audio, aux]
                    v = pyarrow.serialize(v).to_buffer()
                    txn.put(k, v)
                    self.n_out_samples += 1

    @staticmethod
    def get_words_in_time_range(word_list, start_time, end_time):
        words = []

        for word in word_list:
            _, word_s, word_e = word[0], word[1], word[2]

            if word_s >= end_time:
                break

            if word_e <= start_time:
                continue

            words.append(word)

        return words
