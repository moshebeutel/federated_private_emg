import logging
import os
import torch
from common.config import Config
from common.utils import get_exp_name, config_logger


def append_tensors_to_lists(src_folder, dataset_xy_lists: dict):
    for dataset in dataset_xy_lists:
        x_list, y_list = dataset_xy_lists[dataset]
        X = torch.load(os.path.join(src_folder, f'X_{dataset}_windowed.pt'))
        y = torch.load(os.path.join(src_folder, f'y_{dataset}_windowed.pt'))
        x_list.append(X)
        y_list.append(y.reshape(-1, 1))


def main():
    exp_name = get_exp_name(os.path.basename(__file__)[:-3])
    logger = config_logger(f'{exp_name}_logger',
                           level=logging.INFO, log_folder='../log/')
    logger.info(exp_name)
    users_list = ['03', '04', '05']
    x_train_list, y_train_list, x_val_list, y_val_list, x_test_list, y_test_list = [], [], [], [], [], []
    dataset_xy_lists = {'train': (x_train_list, y_train_list),
                        'validation': (x_val_list, y_val_list),
                        'test': (x_test_list, y_test_list)}
    for i, u in enumerate(users_list):
        logger.info(f'Append tensors for user {u}. No. {i} out of {len(users_list)}')
        src_folder = os.path.join(Config.WINDOWED_DATA_DIR, u)
        append_tensors_to_lists(src_folder=src_folder, dataset_xy_lists=dataset_xy_lists)

    for dataset in dataset_xy_lists:
        logger.info(f'Save {dataset} unified tensors')
        x_list, y_list = dataset_xy_lists[dataset]
        torch.save(torch.vstack(x_list), os.path.join(Config.WINDOWED_DATA_DIR, f'X_{dataset}_windowed'))
        torch.save(torch.vstack(y_list), os.path.join(Config.WINDOWED_DATA_DIR, f'y_{dataset}_windowed'))

    logger.info(f'Finished saving  unified tensors')


if __name__ == '__main__':
    main()
