import logging
import os
import numpy as np
import pandas as pd
import torch

from common.config import Config
from common.utils import prepare_X_y_from_dataframe
from common.utils import config_logger, get_exp_name


def main():
    exp_name = get_exp_name(os.path.basename(__file__)[:-3])
    logger = config_logger(f'{exp_name}_logger',
                           level=logging.INFO, log_folder='../log/')
    logger.info(f'{exp_name}... Reading files from {Config.UNIFIED_DATA_DIR}')

    users = os.listdir(Config.UNIFIED_DATA_DIR)
    for u in users:
        hdf_folder = os.path.join(Config.UNIFIED_DATA_DIR, u)
        tensors_folder = os.path.join(Config.TENSORS_DATA_DIR, u)
        save_df_to_tensors(hdf_folder, tensors_folder, ['train', 'validation', 'test'], logger)


def save_df_to_tensors(df_folder, tensors_folder, datasets, logger):
    os.makedirs(tensors_folder, exist_ok=True)
    for dataset in datasets:
        save_df_to_tensor_single_dataset(df_folder, tensors_folder, dataset, logger)


def save_df_to_tensor_single_dataset(df_folder, tensors_folder, dataset, logger):
    hdf_filename = os.path.join(df_folder, f'{dataset}.hdf')
    if not os.path.exists(hdf_filename):
        logger.warning(f'{hdf_filename} does not exist')
    else:
        df = pd.read_hdf(hdf_filename)
        logger.info(f'Loaded train dataframe with {df.shape[0]} rows and columns {df.columns}')
        X, y = prepare_X_y_from_dataframe(df, drop_cols=True)
        logger.info(f'Prepared X, y with X.shape {X.shape} y.shape {y.shape}')
        with open(os.path.join(tensors_folder, f'X_{dataset}.npy'), 'wb') as f:
            np.save(f, X)
        logger.info(f'Finished saving X_{dataset}.npy')
        with open(os.path.join(tensors_folder, f'y_{dataset}.npy'), 'wb') as f:
            np.save(f, y)
        logger.info(f'Finished saving y_{dataset}.npy')
        torch.save(torch.from_numpy(X), os.path.join(tensors_folder, f'X_{dataset}.pt'))
        logger.info(f'Finished saving X_{dataset}.pt')
        torch.save(torch.from_numpy(y), os.path.join(tensors_folder, f'y_{dataset}.pt'))
        logger.info(f'Finished saving y_{dataset}.pt')


if __name__ == '__main__':
    main()
