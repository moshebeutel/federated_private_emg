from __future__ import annotations
import inspect
import os
from datetime import datetime
from enum import Enum
from math import sqrt, log


class Config:
    WORKDIR = os.getcwd()
    LOG_FOLDER = os.path.join(WORKDIR, 'log')
    HDF_FILES_DIR = '../putemg-downloader/Data-HDF5'
    FEATURES_DATAFRAMES_DIR = 'features_dataframes'
    USE_GROUPNORM = False
    USE_DROPOUT = False
    DEVICE = 'cpu'
    BATCH_SIZE = 512
    NUM_EPOCHS = 100
    DETERMINISTIC_SEED = True
    SEED = 42 if DETERMINISTIC_SEED else int(datetime.now().timestamp())

    EARLY_STOP_INCREASING_LOSS_COUNT = 10
    NUM_WORKERS = 1
    OPTIMIZER_TYPE = 'sgd'
    LEARNING_RATE = 0.05

    WEIGHT_DECAY = 1e-3
    MOMENTUM = 0.9
    UNIFIED_DATA_DIR = '../data/unified_dataframes/every10'
    TENSORS_DATA_DIR = '../data/tensors_datasets/every10'
    WINDOWED_DATA_DIR = os.path.join(WORKDIR, 'data/windowed_tensors_datasets/every10')
    NUM_CLASSES = 7
    WINDOW_SIZE = 260
    EVAL_EVERY = 1
    TEST_AT_END = True

    DATASET_TYPE = Enum('DATASET_TYPE', ['putEMG', 'TOY_STORY', 'CIFAR10'])
    DATASET = DATASET_TYPE.CIFAR10

    WRITE_TO_WANDB = False

    # GP
    USE_GP = True
    GP_KERNEL_FUNCTION = 'RBFKernel'
    assert GP_KERNEL_FUNCTION in ['RBFKernel', 'LinearKernel', 'MaternKernel'], \
        f'GP_KERNEL_FUNCTION={GP_KERNEL_FUNCTION} and should be one of RBFKernel, LinearKernel, MaternKernel'
    GP_NUM_GIBBS_STEPS_TRAIN = 5
    GP_NUM_GIBBS_DRAWS_TRAIN = 20
    GP_NUM_GIBBS_STEPS_TEST = 5
    GP_NUM_GIBBS_DRAWS_TEST = 30
    GP_OUTPUTSCALE_INCREASE = 'constant'
    assert GP_OUTPUTSCALE_INCREASE in ['constant', 'increase', 'decrease'], \
        f'GP_OUTPUTSCALE_INCREASE={GP_OUTPUTSCALE_INCREASE} and should be one of constant, increase, decrease'
    GP_OUTPUTSCALE = 8.0
    GP_LENGTHSCALE = 1.0
    GP_PREDICT_RATIO = 0.5
    GP_OBJECTIVE = 'predictive_likelihood'
    assert GP_OBJECTIVE in ['predictive_likelihood', 'marginal_likelihood'], \
        f'GP_OBJECTIVE={GP_OBJECTIVE} and should be one of predictive_likelihood, marginal_likelihood '

    # DP
    DP_METHOD_TYPE = Enum('DP_METHOD_TYPE', ['NO_DP', 'SGD_DP', 'GEP'])
    DP_METHOD = DP_METHOD_TYPE.NO_DP
    USE_GEP = (DP_METHOD == DP_METHOD_TYPE.GEP)
    PUBLIC_USERS_CONTRIBUTE_TO_LEARNING = True
    USE_SGD_DP = (DP_METHOD == DP_METHOD_TYPE.SGD_DP)
    ADD_DP_NOISE = True
    LOT_SIZE_IN_BATCHES = 5
    DP_EPSILON = 1.0
    DP_DELTA = 1e-5

    # FL
    NUM_CLIENTS_PUBLIC, NUM_CLIENT_AGG, NUM_CLIENTS_TRAIN = 4, 46, 400
    assert NUM_CLIENTS_TRAIN >= NUM_CLIENT_AGG, \
        f'Cant aggregate {NUM_CLIENT_AGG} out of {NUM_CLIENTS_TRAIN} train users'
    assert NUM_CLIENTS_TRAIN >= NUM_CLIENTS_PUBLIC, f'Public users can not be more than train users'
    if not USE_GEP:
        NUM_CLIENT_AGG += NUM_CLIENTS_PUBLIC

    NUM_CLIENTS_VAL = 50
    NUM_CLIENTS_TEST = 50
    SAMPLE_CLIENTS_WITH_REPLACEMENT = True
    NUM_INTERNAL_EPOCHS = 1
    CIFAR10_CLASSES_PER_USER = 2


    # GEP
    GEP_NUM_BASES = 10
    GEP_CLIP0 = 0.01  # 10.0  # 50
    GEP_CLIP1 = 0.002  # 20
    # GEP_SIGMA0 = 2.0 * GEP_CLIP0 * sqrt(2.0 * log(1/DP_DELTA))/DP_EPSILON
    # GEP_SIGMA1 = 2.0 * GEP_CLIP1 * sqrt(2.0 * log(1/DP_DELTA))/DP_EPSILON
    GEP_SIGMA0 = 2.0 * sqrt(2.0 * log(1 / DP_DELTA)) / DP_EPSILON
    GEP_SIGMA1 = 2.0 * sqrt(2.0 * log(1 / DP_DELTA)) / DP_EPSILON
    GEP_POWER_ITER = 1
    GEP_NUM_GROUPS = 50
    GEP_USE_RESIDUAL = True

    # DP_SGD
    DP_C = 0.01  # sqrt(pow(GEP_CLIP0, 2.0) + pow(GEP_CLIP1, 2.0))
    DP_SIGMA = 0.0  # sqrt(2 * log(1.25 / DP_DELTA))/DP_EPSILON   # 0.1 * 3.776479532659047  # sqrt(2 * log(1.25 / DP_DELTA))/

    # CIFAR10
    CIFAR10_DATA = (DATASET == DATASET_TYPE.CIFAR10)
    CIFAR10_DATASET_DIR = os.path.join(WORKDIR, 'data/cifar10')

    # TOY STORY
    TOY_STORY = (DATASET == DATASET_TYPE.TOY_STORY)
    INTERNAL_BENCHMARK = False
    PLOT_GRADS = False
    DATA_SCALE = 1.0
    USER_BIAS_SCALE = 0.5 * DATA_SCALE
    DATA_NOISE_SCALE = 0.1 * DATA_SCALE
    DATA_DIM = 24
    HIDDEN_DIM = 50
    OUTPUT_DIM = 7
    GEP_PUBLIC_DATA_SIZE = BATCH_SIZE * 4
    PRIVATE_TRAIN_DATA_SIZE = BATCH_SIZE * 4
    GLOBAL_LEARNING_RATE = LEARNING_RATE  # (PRIVATE_TRAIN_DATA_SIZE / BATCH_SIZE) * LEARNING_RATE

    @staticmethod
    def to_dict() -> dict:
        members = inspect.getmembers(Config, lambda a: not (inspect.isroutine(a)))
        members = [(k, v) for (k, v) in members if not k.startswith('_')]
        return dict(members)
