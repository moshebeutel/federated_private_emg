import logging
import os
import time
import sys
import pandas as pd
import re
import errno
import matplotlib.pyplot as plt

COLS_TO_DROP = ['TRAJ_1', 'type', 'subject', 'trajectory', 'date_time', 'TRAJ_GT_NO_FILTER', 'VIDEO_STAMP']
FULL_USER_LIST = ['03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                  '19', '20', '22', '23', '24', '25', '26', '27', '29', '30', '31', '33', '34', '35', '36', '38',
                  '39', '42', '43', '45', '46', '47', '48', '49', '50', '51', '53', '54']
FULL_TRAJ_LIST = ['sequential', 'repeats_long', 'repeats_short']
HDF_FILES_DIR = '../putemg-downloader/Data-HDF5'
FEATURES_DATAFRAMES_DIR = 'features_dataframes'
TD_FEATURES = ['WL', 'SSC', 'ZC', 'MAV']


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
    return created_logger


def show_learning_curve(train_loss_list, val_loss_list, train_accuracy, val_accuracy,
                        num_epochs, title, figsize=(12, 12)):
    fig, axes = plt.subplots(1, 2, figsize=figsize);
    axes[0].set_xlabel('epochs')
    axes[0].set_ylabel('loss')
    axes[0].plot(range(num_epochs), train_loss_list, label="Train", color='blue')
    axes[0].plot(range(num_epochs), val_loss_list, label="Validation", color='red')
    axes[0].legend()
    axes[0].set_title('Loss vs Epoch')

    axes[1].set_xlabel('epochs')
    axes[1].set_ylabel('accuracy')  # we already handled the x-label with ax1
    axes[1].plot(range(num_epochs), train_accuracy, label="Train", color='blue')
    axes[1].plot(range(num_epochs), val_accuracy, label="Validation", color='red')
    axes[1].legend()
    axes[1].set_title('Accuracy vs Epoch')

    fig.suptitle(title)
    plt.show()


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


def prepare_X_y_from_dataframe(df: pd.DataFrame, target='TRAJ_GT'):
    # remove "idle" and "relax"
    df = df[df[target] > 0]
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


def read_trial(trial: str) -> pd.DataFrame:
    assert type(trial) == str, f'Got bad argument type - {type(trial)}'
    assert trial, 'Got an empty trial name'
    filename_trial = os.path.join(HDF_FILES_DIR, trial)
    assert os.path.exists(filename_trial), f'filename {filename_trial} does not exist'
    record = pd.read_hdf(filename_trial)
    return record
