import logging
import os

import pandas as pd

from common import utils
from common.utils import get_users_list_from_dir, \
    get_traj_for_user, read_trial, read_every_mean_fn
from common.utils import config_logger
# import putemg_features
# from putemg_features import biolab_utilities


DATA_DIR = '/home/moshe/GIT/federated_private_emg/data/Data-HDF5' # '/home/user1/data/datasets/putEMG'  # '../data/reduced_dataframes'
UNIFIED_DATA_DIR = '../data/unified_dataframes/every10'
READ_EVERY = 10
NUM_USERS = len(utils.FULL_USER_LIST)

def is_test(user, traj, is_first_trial):
    return (user, traj) in utils.TEST_SET and is_first_trial


def is_validation(user, traj, is_first_trial):
    return (user, traj) in utils.VALIDATION_SET and not is_first_trial


def train_val_test(user, traj, is_first):
    return 'train' if (not is_validation(user, traj, is_first) and not is_test(user, traj, is_first)) \
        else ('validation' if is_validation(user, traj, is_first) else 'test')


def create_dfs_for_user(u, traj_list, output_fn):
    user_sets_dict = {'train': [], 'validation': [], 'test': []}
    for traj in traj_list:
        for i, tr in enumerate(get_traj_for_user(dir_path=DATA_DIR, traj=traj, user=u)):
            user_sets_dict[train_val_test(u, traj, is_first=(not i))] \
                .append(read_trial(tr, hdf_folder=DATA_DIR,
                                   read_every=READ_EVERY, read_every_fn=read_every_mean_fn, drop_cols=True))
        output_fn(f'Read dataframes for user {u} trajectory {traj}')
    df_user = {}
    os.makedirs(os.path.join(UNIFIED_DATA_DIR, u), exist_ok=True)
    for dataset in ['train', 'validation', 'test']:
        if user_sets_dict[dataset]:
            df_user[dataset] = pd.concat(user_sets_dict[dataset], ignore_index=True, sort=False)
            df_user[dataset].to_hdf(os.path.join(UNIFIED_DATA_DIR, u, f'{dataset}.hdf'), key='df', mode='w')
            del df_user[dataset]
        output_fn(f'Saved dataframes for user {u} dataset {dataset}')
    del df_user


def main():
    exp_name = os.path.basename(__file__)
    logger = config_logger(f'{exp_name}_logger',
                           level=logging.INFO, log_folder='../log/')
    logger.info(f'{exp_name}... Reading files from {DATA_DIR}')
    users_list = get_users_list_from_dir(DATA_DIR)
    traj_list = utils.FULL_TRAJ_LIST
    logger.info(f'users in dir {DATA_DIR}:  {users_list}')
    if len(users_list) > NUM_USERS:
        users_list = users_list[:NUM_USERS]
        logger.info(f'Using {NUM_USERS} users:  {users_list}')

    trials = [f'emg_gestures-{user}-{traj}' for user in users_list for traj in utils.FULL_TRAJ_LIST]

    logger.info(f'Trials in dir {DATA_DIR}:  {trials}')

    # dfs_train, dfs_validation, dfs_test = [], [], []

    for i, u in enumerate(users_list):
        logger.info(f'User number {i}: {u}')
        create_dfs_for_user(u, traj_list, logger.info)

    # test_df = pd.concat(dfs_test, ignore_index=True, sort=False)
    # del dfs_test
    # test_df.to_hdf(os.path.join(UNIFIED_DATA_DIR, 'test.hdf'), key='df', mode='w')
    # del test_df
    # logger.info(f'Saved test df to hdf')
    #
    # train_df = pd.concat(dfs_train, ignore_index=True, sort=False)
    # del dfs_train
    # train_df.to_hdf(os.path.join(UNIFIED_DATA_DIR, 'train.hdf'), key='df', mode='w')
    # del train_df
    # logger.info(f'Saved train df to hdf')

    logger.info('Finished saving dataframes')


if __name__ == '__main__':
    main()
