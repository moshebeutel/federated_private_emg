from __future__ import annotations

import errno
import logging
import os
import re
import sys
import time
import typing
from collections import defaultdict
from datetime import datetime
from math import sqrt, pow
import random

import numpy as np
import pandas as pd
import torch
from common.config import Config

if Config.CIFAR10_DATA:
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import transforms

COLS_TO_DROP = ['TRAJ_1', 'type', 'subject', 'trajectory', 'date_time', 'TRAJ_GT_NO_FILTER', 'VIDEO_STAMP']
FULL_USER_LIST = ['03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                  '19', '20', '22', '23', '24', '25', '26', '27', '29', '30', '31', '33', '34', '35', '36', '38',
                  '39', '42', '43', '45', '46', '47', '48', '49', '50', '51', '53', '54']
FULL_TRAJ_LIST = ['sequential', 'repeats_long', 'repeats_short']
TD_FEATURES = ['WL', 'SSC', 'ZC', 'MAV']

CONCAT_TRAJ = FULL_TRAJ_LIST * int(len(FULL_USER_LIST) / len(FULL_TRAJ_LIST))
VALIDATION_SET = list(zip(FULL_USER_LIST, CONCAT_TRAJ))
TEST_SET = list(zip(FULL_USER_LIST, CONCAT_TRAJ[1:]))

# TOY STORY
DATA_COEFFS = torch.rand((Config.OUTPUT_DIM, Config.DATA_DIM), dtype=torch.float,
                         requires_grad=False) * Config.DATA_SCALE

if Config.TOY_STORY or Config.CIFAR10_DATA:
    train_user_list = [('%d' % i).zfill(3) for i in range(Config.NUM_CLIENTS_PUBLIC + 1, Config.NUM_CLIENTS_TRAIN + 1)]
    public_users = [('%d' % i).zfill(3) for i in range(1, Config.NUM_CLIENTS_PUBLIC + 1)]
    validation_user_list = [('%d' % i).zfill(3) for i in range(Config.NUM_CLIENTS_TRAIN + 1,
                                                               Config.NUM_CLIENTS_TRAIN + Config.NUM_CLIENTS_VAL + 1)]
    test_user_list = [('%d' % i).zfill(3) for i in range(Config.NUM_CLIENTS_TRAIN + Config.NUM_CLIENTS_VAL + 1,
                                                         Config.NUM_CLIENTS_TRAIN + Config.NUM_CLIENTS_VAL +
                                                         Config.NUM_CLIENTS_TEST + 1)]
else:
    public_users = ['04', '13']
    train_user_list = ['04', '13', '35', '08']
    validation_user_list = ['22', '23']
    test_user_list = ['07', '12']

all_users_list = public_users + train_user_list + validation_user_list + test_user_list
USERS_BIASES = {user: bias for (user, bias) in
                zip(all_users_list, (Config.USER_BIAS_SCALE * torch.randn(size=(len(all_users_list),))).tolist())}
USERS_VARIANCES = {user: variance for (user, variance) in zip(all_users_list,
                                                              (Config.DATA_NOISE_SCALE * torch.rand(
                                                                  size=(len(all_users_list),))).tolist())}
if Config.CIFAR10_DATA:
    normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([transforms.ToTensor(), normalization])
    dataset = CIFAR10(
        root=Config.CIFAR10_DATASET_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_set = CIFAR10(
        root=Config.CIFAR10_DATASET_DIR,
        train=False,
        download=True,
        transform=transform
    )

    val_size = len(test_set)  # 10000
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    CIFAR10_TRAIN_LOADER = torch.utils.data.DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True)
    CIFAR10_VALIDATION_LOADER = torch.utils.data.DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=False)
    CIFAR10_TEST_LOADER = torch.utils.data.DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=False)
    CIFAR10_LOADERS = {'train': CIFAR10_TRAIN_LOADER, 'validation': CIFAR10_VALIDATION_LOADER,
                       'test': CIFAR10_TEST_LOADER}

    CIFAR10_USER_LOADERS = {}


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


def labels_from_consecutive(labels):
    labels += 1
    labels[labels > 3] += 2
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


def get_exp_name(module_name: str):
    n = datetime.now()
    time_str = f'_{n.year}_{n.month}_{n.day}_{n.hour}_{n.minute}_{n.second}'
    exp_name = module_name + time_str
    return exp_name


@torch.no_grad()
def create_toy_data(data_size: int, bias: float = 0.0, variance: float = Config.DATA_NOISE_SCALE):
    assert Config.TOY_STORY, 'TOY_STORY disabled'
    X = torch.rand(data_size, 1, Config.DATA_DIM).float()
    y = torch.matmul(X, DATA_COEFFS.T).squeeze()
    y += torch.ones_like(y) * bias
    y += torch.randn_like(y) * variance
    return X, y


def create_cifar_data(datasize, dataset: str):
    raise Exception
    assert dataset in ['train', 'validation', 'test'], 'dataset arguemt should be one of [train, validation, test]'

    num_batches = int(datasize / Config.BATCH_SIZE)
    assert dataset in CIFAR10_LOADERS, 'No cifar loader for ' + dataset
    loader = CIFAR10_LOADERS[dataset]
    X_list, y_list = [], []
    for _ in range(num_batches):
        X, y = next(iter(loader))
        X_list.append(X)
        y_list.append(y.unsqueeze(dim=1))
    del loader

    return torch.vstack(X_list), torch.vstack(y_list).squeeze()


def load_datasets(datasets_folder_name, x_filename, y_filename, datasize=-1, dataset='train', exclude_labels=[]):
    if Config.TOY_STORY:
        u = datasets_folder_name[-3:]
        bias = USERS_BIASES[u]
        variance = USERS_VARIANCES[u]
        # print(f'Create toy data for user {u} who has bias {bias} and variance {variance}')
        assert datasize > 0
        X, y = create_toy_data(data_size=datasize, bias=bias, variance=variance)
    elif Config.CIFAR10_DATA:
        u = datasets_folder_name[-3:]
        assert datasize > 0
        X, y = create_cifar_data(datasize=datasize, dataset=dataset)
        # print(X.shape, y.shape)
    else:
        if isinstance(datasets_folder_name, str):
            datasets_folder_name = [datasets_folder_name]
        X_list, y_list = [], []
        for folder_name in datasets_folder_name:
            labels = torch.load(os.path.join(folder_name, y_filename))
            non_zero_indices = (labels != 0).nonzero(as_tuple=True)[0].long().tolist()
            X_list.append(torch.load(os.path.join(folder_name, x_filename))[non_zero_indices])
            y_list.append(labels[non_zero_indices].unsqueeze(dim=-1))
        X = torch.vstack(X_list)
        y = torch.vstack(y_list).squeeze()
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


def init_data_loaders(datasets_folder_name,
                      datasize=-1,
                      datasets=['train', 'validation', 'test'],
                      batch_size=Config.BATCH_SIZE,
                      num_workers=Config.NUM_WORKERS,
                      output_fn=print):
    loaders = []
    assert datasets, 'Given empty datasets list'
    for dataset in datasets:
        if Config.CIFAR10_DATA:
            u = datasets_folder_name[-3:]
            loader = CIFAR10_USER_LOADERS[u][dataset]
        else:
            X, y = load_datasets(datasets_folder_name, f'X_{dataset}_windowed.pt', f'y_{dataset}_windowed.pt',
                                 datasize=datasize)
            if not Config.TOY_STORY and not Config.CIFAR10_DATA:
                assert_loaded_datasets(X, y)
            output_fn(f'Loaded {dataset} X shape {X.shape}  y shape {y.shape}')
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X, y),
                shuffle=True if dataset == 'train' else False,
                batch_size=batch_size,
                num_workers=num_workers
            )
            del X, y
        loaders.append(loader)
        output_fn(f'Created {dataset} loader. Len = {len(loader)}')

    assert len(loaders) == len(datasets), f'Given {len(datasets)}, Created {len(loaders)} loaders'
    return tuple(loaders) if len(loaders) > 1 else loaders[0]


def assert_loaded_datasets(train_x, train_y):
    assert train_x.shape[1] == Config.WINDOW_SIZE, f'Expected windowed data with window size {Config.WINDOW_SIZE}.' \
                                                   f' Got {train_x.shape[1]}'
    assert train_x.shape[0] >= Config.BATCH_SIZE, f'Batch size is greater than dataset. ' \
                                                  f'Batch size {Config.BATCH_SIZE}, Dataset {train_x.shape[0]}'
    assert train_x.shape[0] == train_y.shape[0], f'Found {train_y.shape[0]} labels for dataset size {train_x.shape[0]}'
    assert train_y.dim() == 1 or train_y.shape[1] == 1, f'Labels expected to have one dimension'


def calc_grad_norm(model):
    grad_norm = 0
    for p in model.parameters():
        # Sum grad squared norms
        grad_norm += pow(float(torch.linalg.vector_norm(p.grad)), 2.0)
    return sqrt(grad_norm)


def flatten_tensor(tensor_list):
    '''
    Taken from https://github.com/dayu11/Gradient-Embedding-Perturbation
    '''
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param


def get_num_classes_samples(dataset):
    """
    Taken from https://github.com/AvivSham/pFedHN.git


    extracts info about certain dataset
    :param dataset: pytorch dataset object
    :return: dataset info number of classes, number of samples, list of labels
    """
    # ---------------#
    # Extract labels #
    # ---------------#
    if isinstance(dataset, torch.utils.data.Subset):
        if isinstance(dataset.dataset.targets, list):
            data_labels_list = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            data_labels_list = dataset.dataset.targets[dataset.indices]
    else:
        if isinstance(dataset.targets, list):
            data_labels_list = np.array(dataset.targets)
        else:
            data_labels_list = dataset.targets
    classes, num_samples = np.unique(data_labels_list, return_counts=True)
    num_classes = len(classes)
    return num_classes, num_samples, data_labels_list


def gen_classes_per_node(dataset, num_users, classes_per_user=2, high_prob=0.6, low_prob=0.4):
    """
    Taken from https://github.com/AvivSham/pFedHN.git


    creates the data distribution of each client
    :param dataset: pytorch dataset object
    :param num_users: number of clients
    :param classes_per_user: number of classes assigned to each client
    :param high_prob: highest prob sampled
    :param low_prob: lowest prob sampled
    :return: dictionary mapping between classes and proportions, each entry refers to other client
    """
    num_classes, num_samples, _ = get_num_classes_samples(dataset)

    # -------------------------------------------#
    # Divide classes + num samples for each user #
    # -------------------------------------------#
    # assert (classes_per_user * num_users) % num_classes == 0, "equal classes appearance is needed"
    count_per_class = (classes_per_user * num_users) // num_classes + 1
    class_dict = {}
    for i in range(num_classes):
        # sampling alpha_i_c
        probs = np.random.uniform(low_prob, high_prob, size=count_per_class)
        # normalizing
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

    # -------------------------------------#
    # Assign each client with data indexes #
    # -------------------------------------#
    class_partitions = defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user):
            class_counts = [class_dict[i]['count'] for i in range(num_classes)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])
    return class_partitions


def gen_data_split(dataset, num_users, class_partitions):
    """
    Taken from https://github.com/AvivSham/pFedHN.git


    divide data indexes for each client based on class_partition
    :param dataset: pytorch dataset object (train/val/test)
    :param num_users: number of clients
    :param class_partitions: proportion of classes per client
    :return: dictionary mapping client to its indexes
    """
    num_classes, num_samples, data_labels_list = get_num_classes_samples(dataset)

    # -------------------------- #
    # Create class index mapping #
    # -------------------------- #
    data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}

    # --------- #
    # Shuffling #
    # --------- #
    for data_idx in data_class_idx.values():
        random.shuffle(data_idx)

    # ------------------------------ #
    # Assigning samples to each user #
    # ------------------------------ #
    user_data_idx = [[] for i in range(num_users)]
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(num_samples[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]

    return user_data_idx


def gen_random_loaders(num_users, bz, classes_per_user):
    """
    Taken from https://github.com/AvivSham/pFedHN.git




    generates train/val/test loaders of each client

    :param num_users: number of clients
    :param bz: batch size
    :param classes_per_user: number of classes assigned to each client
    :return: train/val/test loaders of each client, list of pytorch dataloaders
    """
    loader_params = {"batch_size": bz, "shuffle": False, "pin_memory": True, "num_workers": 0}
    dataloaders = []
    # datasets = get_datasets(data_name, data_path, normalize=True)
    datasets = [train_set, val_set, test_set]
    for i, d in enumerate(datasets):
        # ensure same partition for train/test/val
        if i == 0:
            # train set
            cls_partitions = gen_classes_per_node(d, num_users, classes_per_user)
            loader_params['shuffle'] = True
        usr_subset_idx = gen_data_split(d, num_users, cls_partitions)
        # create subsets for each client
        subsets = list(map(lambda x: torch.utils.data.Subset(d, x), usr_subset_idx))
        # create dataloaders from subsets
        dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))

    return dataloaders
