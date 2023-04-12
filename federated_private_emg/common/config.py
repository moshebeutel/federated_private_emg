from __future__ import annotations
import inspect
from math import sqrt, log


class Config:
    LOG_FOLDER = '/home/moshe/GIT/federated_private_emg/log'
    HDF_FILES_DIR = '../putemg-downloader/Data-HDF5'
    FEATURES_DATAFRAMES_DIR = 'features_dataframes'
    # DEVICE = 'cpu'
    USE_GROUPNORM = False
    USE_DROPOUT = False
    DEVICE = 'cpu'
    BATCH_SIZE = 64
    NUM_EPOCHS = 1000
    EARLY_STOP_INCREASING_LOSS_COUNT = 10
    NUM_WORKERS = 2
    OPTIMIZER_TYPE = 'sgd'
    LEARNING_RATE = 0.001

    WEIGHT_DECAY = 1e-3
    MOMENTUM = 0.9
    UNIFIED_DATA_DIR = '../data/unified_dataframes/every10'
    TENSORS_DATA_DIR = '../data/tensors_datasets/every10'
    WINDOWED_DATA_DIR = '/home/moshe/GIT/federated_private_emg/data/windowed_tensors_datasets/every10'
    NUM_CLASSES = 7
    WINDOW_SIZE = 260
    EVAL_EVERY = 1
    TEST_AT_END = True

    WRITE_TO_WANDB = False

    # FL
    NUM_CLIENT_AGG = 7
    NUM_INTERNAL_EPOCHS = 1


    # DP
    USE_GEP = False
    USE_SGD_DP = True
    ADD_DP_NOISE = True
    LOT_SIZE_IN_BATCHES = 5
    DP_EPSILON = 8.0
    DP_DELTA = 1e-5


    # GEP
    GEP_NUM_BASES = 1
    GEP_CLIP0 = 0.1 #50
    GEP_CLIP1 = 0.02 #20
    GEP_SIGMA0 = 2.0 * GEP_CLIP0 * sqrt(2.0 * log(1/DP_DELTA))/DP_EPSILON
    GEP_SIGMA1 = 2.0 * GEP_CLIP1 * sqrt(2.0 * log(1/DP_DELTA))/DP_EPSILON
    GEP_POWER_ITER = 1
    GEP_NUM_GROUPS = 3

    # DP_SGD
    DP_C = 1000.0 #  sqrt(pow(GEP_CLIP0, 2.0) + pow(GEP_CLIP1, 2.0))
    DP_SIGMA = 0.0  # sqrt(2 * log(1.25 / DP_DELTA))/DP_EPSILON   # 0.1 * 3.776479532659047  # sqrt(2 * log(1.25 / DP_DELTA))/

    # TOY STORY
    TOY_STORY = False
    INTERNAL_BENCHMARK = False
    PLOT_GRADS = False
    DATA_SCALE = 10.0
    USER_BIAS_SCALE = 0.0 * DATA_SCALE
    DATA_NOISE_SCALE = 0.0 * DATA_SCALE
    DATA_DIM = 20
    HIDDEN_DIM = 20
    OUTPUT_DIM = 2
    GEP_PUBLIC_DATA_SIZE = 100
    PRIVATE_TRAIN_DATA_SIZE = BATCH_SIZE * 30
    GLOBAL_LEARNING_RATE = LEARNING_RATE / NUM_CLIENT_AGG # (PRIVATE_TRAIN_DATA_SIZE / BATCH_SIZE) * LEARNING_RATE
    @staticmethod
    def to_dict() -> dict:
        members = inspect.getmembers(Config, lambda a: not (inspect.isroutine(a)))
        members = [(k, v) for (k, v) in members if not k.startswith('_')]
        return dict(members)
