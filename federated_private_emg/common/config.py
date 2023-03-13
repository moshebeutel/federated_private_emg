from __future__ import annotations
import inspect


class Config:
    LOG_FOLDER = '/home/user/GIT/federated_private_emg/log'
    HDF_FILES_DIR = '../putemg-downloader/Data-HDF5'
    FEATURES_DATAFRAMES_DIR = 'features_dataframes'
    # DEVICE = 'cpu'
    USE_GROUPNORM = False
    USE_DROPOUT = False
    DEVICE = 'cuda'
    BATCH_SIZE = 256
    NUM_EPOCHS = 1000
    NUM_WORKERS = 2
    OPTIMIZER_TYPE = 'sgd'
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-3
    MOMENTUM = 0.9
    UNIFIED_DATA_DIR = '../data/unified_dataframes'
    TENSORS_DATA_DIR = '../data/tensors_datasets'
    WINDOWED_DATA_DIR = '/home/user1/GIT/federated_private_emg/data/windowed_tensors_datasets/03'
    NUM_CLASSES = 7
    WINDOW_SIZE = 260
    EVAL_EVERY = 1
    TEST_AT_END = True
    NUM_CLIENT_AGG = 5
    NUM_INTERNAL_EPOCHS = 20
    WRITE_TO_WANDB = True
    # DP
    ADD_DP_NOISE = False
    LOT_SIZE_IN_BATCHES = 5
    DP_C = 0.1
    DP_EPSILON = 1.0
    DP_DELTA = 0.001
    DP_SIGMA = 3.776479532659047  # sqrt(2 * log(1.25 / DP_DELTA))

    @staticmethod
    def to_dict() -> dict:
        members = inspect.getmembers(Config, lambda a: not (inspect.isroutine(a)))
        members = [(k, v) for (k, v) in members if not k.startswith('_')]
        return dict(members)
