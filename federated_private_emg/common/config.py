from __future__ import annotations


class Config:
    LOG_FOLDER = '/home/user/GIT/federated_private_emg/log'
    HDF_FILES_DIR = '../putemg-downloader/Data-HDF5'
    FEATURES_DATAFRAMES_DIR = 'features_dataframes'
    DEVICE = 'cpu'
    BATCH_SIZE = 16
    NUM_EPOCHS = 2
    NUM_WORKERS = 2
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-3
    UNIFIED_DATA_DIR = '../data/unified_dataframes'
    TENSORS_DATA_DIR = '../data/tensors_datasets'
    WINDOWED_DATA_DIR = '../data/windowed_tensors_datasets'
    NUM_CLASSES = 7
    WINDOW_SIZE = 260
    EVAL_EVERY = 1
    NUM_CLIENT_AGG = 5
    NUM_INTERNAL_EPOCHS = 1
    WRITE_TO_WANDB = False
    # DP
    ADD_DP_NOISE = True
    DP_C = 0.001
    DP_EPSILON = 9 / 20
    DP_DELTA = 0.001
    DP_SIGMA = 3.776479532659047  # sqrt(2 * log(1.25 / DP_DELTA))
