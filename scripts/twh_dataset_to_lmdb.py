import argparse
import glob
import os
from pathlib import Path

import librosa
import lmdb
import pyarrow
from sklearn.pipeline import Pipeline
import joblib as jl

from pymo.preprocessing import *
from pymo.parsers import BVHParser
from utils.data_utils import SubtitleWrapper, normalize_string

# 18 joints (only upper body)
target_joints = ['b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']

# 24 joints (upper and lower body excluding fingers)
# target_joints = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']

# 56 joints (upper and lower body including fingers)
# target_joints = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1', 'b_l_index2', 'b_l_index3', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1', 'b_r_index2', 'b_r_index3', 'b_neck0', 'b_head']


def process_bvh(gesture_filename, dump_pipeline=False):
    p = BVHParser()

    data_all = list()
    data_all.append(p.parse(gesture_filename))

    data_pipe = Pipeline([
        # ('dwnsampl', DownSampler(tgt_fps=30, keep_all=False)),
        # ('root', RootNormalizer()),
        ('jtsel', JointSelector(target_joints, include_root=False)),
        # ('mir', Mirror(axis='X', append=True)),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    if dump_pipeline:
        jl.dump(data_pipe, os.path.join('../resource', 'data_pipe.sav'))

    # euler -> rotation matrix
    out_data = out_data.reshape((out_data.shape[0], out_data.shape[1], -1, 6))  # 3 pos (XYZ), 3 rot (ZXY)
    out_matrix = np.zeros((out_data.shape[0], out_data.shape[1], out_data.shape[2], 12))  # 3 pos, 1 rot matrix (9 elements)
    for i in range(out_data.shape[0]):  # mirror
        for j in range(out_data.shape[1]):  # frames
            for k in range(out_data.shape[2]):  # joints
                out_matrix[i, j, k, :3] = out_data[i, j, k, :3]  # positions
                r = R.from_euler('ZXY', out_data[i, j, k, 3:], degrees=True)
                out_matrix[i, j, k, 3:] = r.as_matrix().flatten()  # rotations
    out_matrix = out_matrix.reshape((out_data.shape[0], out_data.shape[1], -1))

    return out_matrix[0]


def make_lmdb_gesture_dataset(base_path):
    gesture_path = os.path.join(base_path, 'bvh')
    audio_path = os.path.join(base_path, 'wav')
    text_path = os.path.join(base_path, 'tsv')
    out_path = os.path.join(base_path, 'lmdb')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    map_size = 1024 * 20  # in MB
    map_size <<= 20  # in B
    db = [lmdb.open(os.path.join(out_path, 'lmdb_train'), map_size=map_size),
          lmdb.open(os.path.join(out_path, 'lmdb_test'), map_size=map_size)]

    # delete existing files
    for i in range(2):
        with db[i].begin(write=True) as txn:
            txn.drop(db[i].open_db())

    all_poses = []
    bvh_files = sorted(glob.glob(gesture_path + "/*.bvh"))
    save_idx = 0
    for bvh_file in bvh_files:
        name = os.path.split(bvh_file)[1][:-4]
        print(name)

        # load subtitles
        tsv_path = os.path.join(text_path, name + '.tsv')
        if os.path.isfile(tsv_path):
            subtitle = SubtitleWrapper(tsv_path).get()
        else:
            continue

        # load audio
        wav_path = os.path.join(audio_path, '{}.wav'.format(name))
        if os.path.isfile(wav_path):
            audio_raw, audio_sr = librosa.load(wav_path, mono=True, sr=16000, res_type='kaiser_fast')
        else:
            continue

        # load skeletons
        dump_pipeline = (save_idx == 2)  # trn_2022_v1_002 has a good rest finger pose
        # poses, poses_mirror = process_bvh(bvh_file)
        poses = process_bvh(bvh_file, dump_pipeline)

        # process
        clips = [{'vid': name, 'clips': []},  # train
                 {'vid': name, 'clips': []}]  # validation

        # split
        if save_idx % 100 == 0:
            dataset_idx = 1  # validation
        else:
            dataset_idx = 0  # train

        # word preprocessing
        word_list = []
        for wi in range(len(subtitle)):
            word_s = float(subtitle[wi][0])
            word_e = float(subtitle[wi][1])
            word = subtitle[wi][2].strip()

            word_tokens = word.split()

            for t_i, token in enumerate(word_tokens):
                token = normalize_string(token)
                if len(token) > 0:
                    new_s_time = word_s + (word_e - word_s) * t_i / len(word_tokens)
                    new_e_time = word_s + (word_e - word_s) * (t_i + 1) / len(word_tokens)
                    word_list.append([token, new_s_time, new_e_time])

        # save subtitles and skeletons
        poses = np.asarray(poses, dtype=np.float16)
        clips[dataset_idx]['clips'].append(
            {'words': word_list,
             'poses': poses,
             'audio_raw': audio_raw
             })
        # poses_mirror = np.asarray(poses_mirror, dtype=np.float16)
        # clips[dataset_idx]['clips'].append(
        #     {'words': word_list,
        #      'poses': poses_mirror,
        #      'audio_raw': audio_raw
        #      })

        # write to db
        for i in range(2):
            with db[i].begin(write=True) as txn:
                if len(clips[i]['clips']) > 0:
                    k = '{:010}'.format(save_idx).encode('ascii')
                    v = pyarrow.serialize(clips[i]).to_buffer()
                    txn.put(k, v)

        all_poses.append(poses)
        save_idx += 1

    # close db
    for i in range(2):
        db[i].sync()
        db[i].close()

    # calculate data mean
    all_poses = np.vstack(all_poses)
    pose_mean = np.mean(all_poses, axis=0, dtype=np.float64)
    pose_std = np.std(all_poses, axis=0, dtype=np.float64)

    print('data mean/std')
    print('data_mean:', str(["{:0.5f}".format(e) for e in pose_mean]).replace("'", ""))
    print('data_std:', str(["{:0.5f}".format(e) for e in pose_std]).replace("'", ""))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", type=Path)
    args = parser.parse_args()

    make_lmdb_gesture_dataset(args.db_path)
