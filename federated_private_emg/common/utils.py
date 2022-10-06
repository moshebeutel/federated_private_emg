from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import re
import errno

import torch
import typing
import wandb
from tqdm import tqdm

from common.config import Config

COLS_TO_DROP = ['TRAJ_1', 'type', 'subject', 'trajectory', 'date_time', 'TRAJ_GT_NO_FILTER', 'VIDEO_STAMP']
FULL_USER_LIST = ['03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                  '19', '20', '22', '23', '24', '25', '26', '27', '29', '30', '31', '33', '34', '35', '36', '38',
                  '39', '42', '43', '45', '46', '47', '48', '49', '50', '51', '53', '54']
FULL_TRAJ_LIST = ['sequential', 'repeats_long', 'repeats_short']
TD_FEATURES = ['WL', 'SSC', 'ZC', 'MAV']

CONCAT_TRAJ = FULL_TRAJ_LIST * int(len(FULL_USER_LIST) / len(FULL_TRAJ_LIST))
VALIDATION_SET = list(zip(FULL_USER_LIST, CONCAT_TRAJ))
TEST_SET = list(zip(FULL_USER_LIST, CONCAT_TRAJ[1:]))


class SimpleGlobalCounter:
    """
    Counter that can be safely passed to functions that increments its value.
    Overcome Python's lack of pass by reference.
    """

    def __init__(self, init_val: int = 0):
        """

        :param init_val: Initial counter value. Default: 0
        """
        self._v = init_val

    def increment(self, by: int = 1):
        """
        Increment counter ++
        :param by: Enable increment by more than 1. Default: 1
        :return: None
        """
        assert by > 0, f'Increment counter should be by positive value. Got {by}'
        self._v += by

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, val):
        self._v = val

    def __str__(self):
        return str(self._v)


def labels_to_consecutive(labels):
    labels[labels > 5] -= 2
    labels -= 1
    return labels


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
    # return traj_files_for_user[0], traj_files_for_user[1]
    return traj_files_for_user


def add_dummy_for_user_trajectory(trial, user, traj, users_list=FULL_USER_LIST, traj_list=FULL_TRAJ_LIST):
    for u in users_list:
        trial[f'u_{u}'] = 0
    for t in traj_list:
        trial[f't_{t}'] = 0
    trial[f'u_{user}'] = 1
    trial[f'u_{traj}'] = 1
    return trial


def read_trial(trial: str, hdf_folder: str = Config.HDF_FILES_DIR, read_every: int = 1,
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
        record_values = record.values[::read_every] if read_every_fn is None \
            else read_every_fn(record.values, read_every)
        record = pd.DataFrame(columns=record.columns, data=record_values)
    return record


def read_every_mean_fn(a: np.array, read_every: int) -> np.array:
    assert read_every < a.shape[0], f'read_every should be less than a.shape[0]. Got read_every = {read_every}'
    return np.array([a[read_every * i:read_every * i + read_every].mean(axis=0)
                     for i in range(int(a.shape[0] // read_every))])


def run_single_epoch(loader: torch.utils.data.DataLoader,
                     model: torch.nn.Module,
                     criterion: torch.nn.CrossEntropyLoss,
                     optimizer: torch.optim.Optimizer = None) -> (float, float):
    running_loss, correct_counter, sample_counter, counter = 0, 0, 0, 0
    y_pred, y_labels = [], []
    for k, batch in enumerate(loader):
        curr_batch_size = batch[1].size(0)
        if curr_batch_size < Config.BATCH_SIZE:
            continue
        counter += 1

        sample_counter += curr_batch_size
        batch = (t.to(Config.DEVICE) for t in batch)
        emg, labels = batch
        labels = labels_to_consecutive(labels)
        if model.training and optimizer is not None:
            optimizer.zero_grad()
        outputs = model(emg.float())

        loss = criterion(outputs, labels.long())
        running_loss += float(loss)
        _, predicted = torch.max(outputs.data, 1)
        if model.training:
            loss.backward()
            if optimizer is not None:
                optimizer.step()
        correct = (predicted == labels).sum().item()
        correct_counter += int(correct)

        y_pred += predicted.cpu().tolist()
        y_labels += labels.cpu().tolist()
    loss = running_loss / float(counter)
    acc = 100 * correct_counter / sample_counter
    return loss, acc


def get_exp_name(module_name: str):
    n = datetime.now()
    time_str = f'_{n.year}_{n.month}_{n.day}_{n.hour}_{n.minute}_{n.second}'
    exp_name = module_name + time_str
    return exp_name


def load_datasets(datasets_folder_name, x_filename, y_filename):
    X = torch.load(os.path.join(datasets_folder_name, x_filename))

    y = torch.load(os.path.join(datasets_folder_name, y_filename))

    return X, y


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


def wandb_log(epoch, test_acc, train_acc, train_loss, test_loss=0):
    if Config.WRITE_TO_WANDB:
        wandb.log({'epoch': epoch,
                   'train_loss': train_loss,
                   'train_acc': train_acc,
                   'test_loss': test_loss,
                   'test_acc': test_acc
                   })


def train_model(criterion, model, optimizer, train_loader, test_loader,
                num_epochs: int = 1, eval_every: int = 1,
                epoch_level_optimization: bool = False,
                add_dp_noise_before_optimization=False,
                log2wandb=True,
                show_pbar=True):
    test_loss, test_acc = 100, 0
    train_loss, train_acc = 100, 0
    eval_num = 0
    epoch_pbar = tqdm(range(num_epochs)) if show_pbar else None
    model.train()
    for epoch in (epoch_pbar if show_pbar else range(num_epochs)):
        if epoch_level_optimization:
            optimizer.zero_grad()  # optimization done at epoch level
        epoch_train_loss, epoch_train_acc = \
            run_single_epoch(loader=train_loader, model=model, criterion=criterion,
                             optimizer=optimizer if not epoch_level_optimization else None
                             )  # Do not pass optimizer if optimization is done once per epoch
        if epoch_level_optimization:
            if add_dp_noise_before_optimization:
                add_dp_noise(model)
            optimizer.step()  # optimization done at epoch level

        if eval_every == 1 or (eval_every > 1 and epoch % (eval_every - 1)) == 0:
            model.eval()
            epoch_test_loss, epoch_test_acc = run_single_epoch(loader=test_loader, model=model, criterion=criterion)
            model.train()
            eval_num += 1
            test_loss += epoch_test_loss
            test_acc += epoch_test_acc
        if show_pbar:
            epoch_pbar.set_description(
                f'epoch {epoch} loss {epoch_train_loss} acc {epoch_train_acc}'
                f' test loss {epoch_test_loss} test acc {epoch_test_acc}')

        if log2wandb:
            wandb_log(epoch, train_loss=train_loss / num_epochs, train_acc=train_acc / num_epochs,
                      test_loss=test_loss / eval_num, test_acc=test_acc / eval_num)

        train_acc += epoch_train_acc
        train_loss += epoch_train_acc

    return train_loss / num_epochs, train_acc / num_epochs, test_loss / eval_num, test_acc / eval_num


def init_data_loaders(datasets_folder_name,
                      x_train_filename='X_train_windowed.pt',
                      y_train_filename='y_train_windowed.pt',
                      x_test_filename='X_test_windowed.pt',
                      y_test_filename='y_test_windowed.pt',
                      output_fn=print):
    train_x, train_y = load_datasets(datasets_folder_name, x_train_filename, y_train_filename)
    assert_loaded_datasets(train_x, train_y)
    output_fn(f'Loaded train_x shape {train_x.shape} train_y shape {train_y.shape}')
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        shuffle=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS
    )
    test_x, test_y = load_datasets(datasets_folder_name, x_test_filename, y_test_filename)
    assert_loaded_datasets(test_x, test_y)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        shuffle=False,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS
    )
    output_fn(f'Test Loader created, Len {len(test_loader)}')
    return test_loader, train_loader


def assert_loaded_datasets(train_x, train_y):
    assert train_x.shape[1] == Config.WINDOW_SIZE, f'Expected windowed data with window size {Config.WINDOW_SIZE}.' \
                                                   f' Got {train_x.shape[1]}'
    assert train_x.shape[0] >= Config.BATCH_SIZE, f'Batch size is greater than dataset. ' \
                                                  f'Batch size {Config.BATCH_SIZE}, Dataset {train_x.shape[0]}'
    assert train_x.shape[0] == train_y.shape[0], f'Found {train_y.shape[0]} labels for dataset size {train_x.shape[0]}'
    assert train_y.dim() == 1 or train_y.shape[1] == 1, f'Labels expected to have one dimension'


def add_dp_noise(model):
    grad_norm = 0
    for p in model.parameters():
        # Sum grid norms
        grad_norm += float(torch.linalg.vector_norm(p.grad))
    for p in model.parameters():
        # Clip gradients
        p.grad /= max(1, grad_norm / Config.DP_C)
        # Add DP noise to gradients
        noise = torch.randn_like(p.grad) * Config.DP_SIGMA * Config.DP_C
        p.grad += noise
        p.grad /= Config.BATCH_SIZE  # Averaging.Use batch as the 'Lot'
