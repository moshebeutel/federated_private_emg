import logging
import os
import utils
import pandas as pd
from federated_private_emg.utils import config_logger, get_users_list_from_dir, \
    get_traj_for_user, add_dummy_for_user_trajectory, read_trial

DATA_DIR = '../data/Data-HDF5'  # '../data/reduced_dataframes'
UNIFIED_DATA_DIR = '../data/unified_dataframes'
READ_EVERY = 50
NUM_USERS = 4

def is_test(idx, traj):
    return (idx, traj) in [
        (0, 'sequential'),
        (1, 'repeats_short'),
        (2, 'repeats_long')
    ]


def main():
    exp_name = os.path.basename(__file__)
    logger = config_logger(f'{exp_name}_logger',
                           level=logging.INFO, log_folder='../log/')
    logger.info(f'{exp_name}... Reading files from {DATA_DIR}')
    users_list = get_users_list_from_dir(DATA_DIR)
    logger.info(f'users in dir {DATA_DIR}:  {users_list}')
    if len(users_list) > NUM_USERS:
        users_list = users_list[:NUM_USERS]
        logger.info(f'Using {NUM_USERS} users:  {users_list}')

    trials = [f'emg_gestures-{user}-{traj}' for user in users_list for traj in utils.FULL_TRAJ_LIST]

    logger.info(f'Trials in dir {DATA_DIR}:  {trials}')

    dfs_train, dfs_test = [], []
    for i, u in enumerate(users_list):
        for traj in utils.FULL_TRAJ_LIST:
            tr1, tr2 = get_traj_for_user(dir_path=DATA_DIR, traj=traj, user=u)
            logger.info(f'Found {traj} Trajectories for user {u}:\n{tr1}\n{tr2}')

            dfs_train.append(read_trial(tr1, hdf_folder=DATA_DIR, read_every=READ_EVERY, drop_cols=True))
            # df.to_hdf(train_df_filename, key='df', mode='a')
            # del df
            df = read_trial(tr1, hdf_folder=DATA_DIR, read_every=READ_EVERY, drop_cols=True)
            if is_test(i, traj):
                dfs_test.append(df)
            else:
                dfs_train.append(df)

            logger.info(f'Read dataframes for user {u} trajectory {traj}')

    test_df = pd.concat(dfs_test, ignore_index=True, sort=False)
    del dfs_test
    test_df.to_hdf(os.path.join(UNIFIED_DATA_DIR, 'test.hdf'), key='df',  mode='w')
    del test_df
    logger.info(f'Saved test df to hdf')

    train_df = pd.concat(dfs_train, ignore_index=True, sort=False)
    del dfs_train
    train_df.to_hdf(os.path.join(UNIFIED_DATA_DIR, 'train.hdf'), key='df', mode='w')
    del train_df
    logger.info(f'Saved train df to hdf')

    logger.info('Finished saving dataframes')


if __name__ == '__main__':
    main()
