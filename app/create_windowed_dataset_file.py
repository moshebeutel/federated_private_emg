import logging
import os
import torch
import numpy as np
from utils import config_logger

TENSORS_DATA_DIR = '../data/tensors_datasets'
WINDOWED_DATA_DIR = '../data/windowed_tensors_datasets'
WINDOW_SIZE = 260
WINDOW_STRIDE = int(WINDOW_SIZE / 2)
STRIDE_RATIO = int(WINDOW_SIZE / WINDOW_STRIDE)


def main():
    exp_name = os.path.basename(__file__)
    logger = config_logger(f'{exp_name}_logger',
                           level=logging.INFO, log_folder='../log/')

    logger.info('Load tensors')
    X_train = torch.load(os.path.join(TENSORS_DATA_DIR, 'X_train.pt'))
    assert X_train.dim() == 2

    logger.info(f'Loaded tensor X_train shape {X_train.shape}')
    max_ind = X_train.shape[0] - (X_train.shape[0] % WINDOW_SIZE)
    logger.debug(f'source {X_train.shape[0]} window {WINDOW_SIZE} max {max_ind}')

    X_train_windowed = X_train[:max_ind, :].reshape(-1, WINDOW_SIZE, X_train.shape[1])
    logger.info(f'Finished windowing X_train_windowed shape {X_train_windowed.shape}')
    torch.save(X_train_windowed, os.path.join(WINDOWED_DATA_DIR, 'X_train_windowed.pt'))
    with open(os.path.join(WINDOWED_DATA_DIR, 'X_train_windowed.npy'), 'wb') as f:
        np.save(f, X_train_windowed.numpy())
    del X_train, X_train_windowed
    logger.info(f'EMG windows saved to file saved to file')

    y_train = torch.load(os.path.join(TENSORS_DATA_DIR, 'y_train.pt'))
    logger.info(f'Loaded tensor y_train shape {y_train.shape}')
    y_train_windowed = y_train[:max_ind].reshape(-1, WINDOW_SIZE)

    logger.info(f'Finished windowing y_train_windowed shape {y_train_windowed.shape}')
    y_train_windowed = y_train_windowed.mean(dim=1).round()
    logger.info(f'Labels mean and round')

    torch.save(y_train_windowed, os.path.join(WINDOWED_DATA_DIR, 'y_train_windowed.pt'))
    with open(os.path.join(WINDOWED_DATA_DIR, 'y_train_windowed.npy'), 'wb') as f:
        np.save(f, y_train_windowed.numpy())
    del y_train, y_train_windowed
    logger.info(f'Labels saved to file')

    logger.info('Load test tensors')
    X_test = torch.load(os.path.join(TENSORS_DATA_DIR, 'X_test.pt'))
    assert X_test.dim() == 2

    logger.info(f'Loaded tensor X_test shape {X_test.shape}')
    max_ind = X_test.shape[0] - (X_test.shape[0] % WINDOW_SIZE)
    logger.debug(f'source {X_test.shape[0]} window {WINDOW_SIZE} max {max_ind}')

    X_test_windowed = X_test[:max_ind, :].reshape(-1, WINDOW_SIZE, X_test.shape[1])
    logger.info(f'Finished windowing X_test_windowed shape {X_test_windowed.shape}')
    torch.save(X_test_windowed, os.path.join(WINDOWED_DATA_DIR, 'X_test_windowed.pt'))
    with open(os.path.join(WINDOWED_DATA_DIR, 'X_test_windowed.npy'), 'wb') as f:
        np.save(f, X_test_windowed.numpy())
    del X_test, X_test_windowed
    logger.info(f'EMG test windows saved to file saved to file')

    y_test = torch.load(os.path.join(TENSORS_DATA_DIR, 'y_test.pt'))
    logger.info(f'Loaded tensor y_test shape {y_test.shape}')
    y_test_windowed = y_test[:max_ind].reshape(-1, WINDOW_SIZE)

    logger.info(f'Finished windowing y_test_windowed shape {y_test_windowed.shape}')
    y_test_windowed = y_test_windowed.mean(dim=1).round()
    logger.info(f'Labels mean and round')

    torch.save(y_test_windowed, os.path.join(WINDOWED_DATA_DIR, 'y_test_windowed.pt'))
    with open(os.path.join(WINDOWED_DATA_DIR, 'y_test_windowed.npy'), 'wb') as f:
        np.save(f, y_test_windowed.numpy())
    del y_test, y_test_windowed
    logger.info(f'Labels saved to file')


if __name__ == '__main__':
    main()
