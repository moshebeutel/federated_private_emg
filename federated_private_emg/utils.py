from __future__ import annotations
import logging
import os
import time
import sys

import numpy as np
import pandas as pd
import re
import errno

COLS_TO_DROP = ['TRAJ_1', 'type', 'subject', 'trajectory', 'date_time', 'TRAJ_GT_NO_FILTER', 'VIDEO_STAMP']
FULL_USER_LIST = ['03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                  '19', '20', '22', '23', '24', '25', '26', '27', '29', '30', '31', '33', '34', '35', '36', '38',
                  '39', '42', '43', '45', '46', '47', '48', '49', '50', '51', '53', '54']
FULL_TRAJ_LIST = ['sequential', 'repeats_long', 'repeats_short']
HDF_FILES_DIR = '../putemg-downloader/Data-HDF5'
FEATURES_DATAFRAMES_DIR = 'features_dataframes'
TD_FEATURES = ['WL', 'SSC', 'ZC', 'MAV']
LOG_FOLDER = '/home/user/GIT/federated_private_emg/log'


def labels_to_consecutive(labels):
    labels[labels > 5] -= 2
    labels -= 1
    return labels


def config_logger(name='default', level=logging.DEBUG, log_folder='./log/'):
    # config logger
    log_format = '%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(message)s'
    formatter = logging.Formatter(log_format)
    logging.basicConfig(level=level,
                        format=log_format,
                        filename=f'{log_folder}{time.ctime()}_{name}.log',
                        filemode='w')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    created_logger = logging.getLogger(name + '_logger')
    created_logger.addHandler(handler)
    logging.getLogger(name).setLevel(level)
    return created_logger


def prepare_X_y(data_file: str, target='TRAJ_GT', drop_cols=True, min_action=0):
    # read dataframe from file
    df = pd.read_hdf(data_file)
    # drop irrelevant columns
    if drop_cols:
        df.drop(COLS_TO_DROP, axis=1, inplace=True)
    # remove "idle" and "relax"
    df = df[df[target] > min_action]
    X = df.iloc[:, 0:24].to_numpy()
    y = df[target].to_numpy()
    del df
    return X, y


def prepare_X_y_from_dataframe(df: pd.DataFrame, target='TRAJ_GT', drop_cols=False):
    # remove "idle" and "relax"
    df = df[df[target] > 0]
    if drop_cols:
        for col in [c for c in COLS_TO_DROP if c in df.columns]:
            df = df.drop(col, axis=1)
    y = df[target].to_numpy()
    X = df.drop([target], axis=1)
    X = X.to_numpy()
    return X, y


def get_users_list_from_dir(dir_path: str):
    assert type(dir_path) == str, f'Got {dir_path} instead of string'
    if not os.path.exists(dir_path):
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), dir_path)
    files = os.listdir(dir_path)
    assert len(files) > 0, f'Got an empty directory {dir_path}'
    reg_traj_user = re.compile(r'emg_gestures-\d\d-repeats_long')
    k = [reg_traj_user.findall(f) for f in files]
    k = [f for f in k if len(f) > 0]
    k = [f[0] for f in k]
    reg_user = re.compile(r'\d\d')
    users = [reg_user.findall(f) for f in k]
    assert all([len(f) == 1 for f in users]), 'Some reg_user returned more than one answer'
    users = [f[0] for f in users]
    users = list(set(users))
    assert len(users) > 0
    return users


def get_traj_for_user(dir_path: str, traj: str, user: str):
    assert type(dir_path) == str, f'Got {dir_path} instead of string'
    assert traj in FULL_TRAJ_LIST, f'traj argument should be one of sequential,' \
                                   f' repeats-long or repeats-short - got {traj}'
    assert user in FULL_USER_LIST
    if not os.path.isdir(dir_path):
        print('{:s} is not a valid folder'.format(dir_path))
        exit(1)

    files = os.listdir(dir_path)
    assert len(files) > 0, f'Got an empty directory {dir_path}'
    traj_files_for_user = [f for f in files if f'emg_gestures-{user}-{traj}' in f]
    assert int(len(traj_files_for_user)) == 2, 'Expected 2 experiments per user per trajectory - got {int(len(' \
                                               'traj_files_for_user))} '
    return traj_files_for_user[0], traj_files_for_user[1]


def add_dummy_for_user_trajectory(trial, user, traj, users_list=FULL_USER_LIST, traj_list=FULL_TRAJ_LIST):
    for u in users_list:
        trial[f'u_{u}'] = 0
    for t in traj_list:
        trial[f't_{t}'] = 0
    trial[f'u_{user}'] = 1
    trial[f'u_{traj}'] = 1
    return trial


def read_trial(trial: str, hdf_folder: str = HDF_FILES_DIR, read_every: int = 1,
               read_every_fn: typing.Callable[[np.array, int], np.array] = None,
               drop_cols: bool = False) -> pd.DataFrame:
    assert type(trial) == str, f'Got bad argument type - {type(trial)}'
    assert trial, 'Got an empty trial name'
    filename_trial = os.path.join(hdf_folder, trial)
    assert os.path.exists(filename_trial), f'filename {filename_trial} does not exist'
    record = pd.read_hdf(filename_trial)
    if drop_cols:
        cols2drop = [c for c in COLS_TO_DROP if c in record.columns]
        record = record.drop(cols2drop, axis=1)
    if read_every > 1:
        record_values =  record.values[::read_every] if read_every_fn is None\
            else read_every_fn(record.values, read_every)
        record = pd.DataFrame(columns=record.columns, data=record_values)
    return record

def read_every_mean_fn(a: np.array, read_every: int) -> np.array:
    assert read_every < a.shape[0], f'read_every should be less than a.shape[0]. Got read_every = {read_every}'
    return np.array([a[read_every*i:read_every*i+read_every].mean(axis=0)
                     for i in range(int(a.shape[0] // read_every))])
