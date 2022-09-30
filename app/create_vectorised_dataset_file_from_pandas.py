import logging
import os
import numpy as np
import pandas as pd
import torch

from federated_private_emg.utils import config_logger, prepare_X_y_from_dataframe

UNIFIED_DATA_DIR = '../data/unified_dataframes'
TENSORS_DATA_DIR = '../data/tensors_datasets'


def main():
    exp_name = os.path.basename(__file__)
    logger = config_logger(f'{exp_name}_logger',
                           level=logging.INFO, log_folder='../log/')
    logger.info(f'{exp_name}... Reading files from {UNIFIED_DATA_DIR}')

    train_df = pd.read_hdf(os.path.join(UNIFIED_DATA_DIR, 'train.hdf'))
    logger.info(f'Loaded train dataframe with {train_df.shape[0]} rows and columns {train_df.columns}')
    X_train, y_train = prepare_X_y_from_dataframe(train_df, drop_cols=True)
    logger.info(f'Prepared X, y with X.shape {X_train.shape} y.shape {y_train.shape}')

    test_df = pd.read_hdf(os.path.join(UNIFIED_DATA_DIR, 'test.hdf'))
    X_test, y_test = prepare_X_y_from_dataframe(test_df)

    with open(os.path.join(TENSORS_DATA_DIR, 'X_test.npy'), 'wb') as f:
        np.save(f, X_test)
    logger.info(f'Finished saving X_test.npy')

    with open(os.path.join(TENSORS_DATA_DIR, 'X_train.npy'), 'wb') as f:
        np.save(f, X_train)
    logger.info(f'Finished saving X_train.npy')
    with open(os.path.join(TENSORS_DATA_DIR, 'y_test.npy'), 'wb') as f:
        np.save(f, y_test)
    logger.info(f'Finished saving y_test.npy')
    with open(os.path.join(TENSORS_DATA_DIR, 'y_train.npy'), 'wb') as f:
        np.save(f, y_train)
    logger.info(f'Finished saving y_train.npy')

    torch.save(torch.from_numpy(X_train), os.path.join(TENSORS_DATA_DIR, 'X_train.pt'))
    logger.info(f'Finished saving X_train.pt')

    torch.save(torch.from_numpy(X_test), os.path.join(TENSORS_DATA_DIR, 'X_test.pt'))
    logger.info(f'Finished saving X_test.pt')

    torch.save(torch.from_numpy(y_train), os.path.join(TENSORS_DATA_DIR, 'y_train.pt'))
    logger.info(f'Finished saving y_train.pt')

    torch.save(torch.from_numpy(y_test), os.path.join(TENSORS_DATA_DIR, 'y_test.pt'))
    logger.info(f'Finished saving y_test.pt')

    logger.info(f'Finished saving files')


if __name__ == '__main__':
    main()
