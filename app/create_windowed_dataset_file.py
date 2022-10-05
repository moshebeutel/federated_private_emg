import logging
import os
from pprint import pprint
import torch
import numpy as np
from common.utils import config_logger
from common.config import Config

WINDOW_SIZE = 260
WINDOW_SHIFT = int(WINDOW_SIZE / 13)
WINDOW_SHIFT_LIST = [i * WINDOW_SHIFT for i in range(1, int(WINDOW_SIZE / WINDOW_SHIFT))]


def main():
    exp_name = os.path.basename(__file__)

    logger = config_logger(f'{exp_name}_logger', level=logging.INFO, log_folder='../log/')

    all_users_tensors_folder = os.listdir(Config.TENSORS_DATA_DIR)
    for u in all_users_tensors_folder:

        src_tensors_folder = os.path.join(Config.TENSORS_DATA_DIR, u)
        windowed_tensors_folder = os.path.join(Config.WINDOWED_DATA_DIR, u)
        os.makedirs(windowed_tensors_folder, exist_ok=True)

        for dataset in ['train', 'validation', 'test']:
            create_windows(src_tensors_folder=src_tensors_folder, windowed_tensors_folder=windowed_tensors_folder,
                           x_filename=f'X_{dataset}', y_filename=f'y_{dataset}', output_fn=logger.info)


def create_windows(src_tensors_folder, windowed_tensors_folder,
                   x_filename: str = 'X_train',
                   y_filename: str = 'y_train',
                   window_shifts=WINDOW_SHIFT_LIST,
                   output_fn=pprint):
    sec_tensor_fname = os.path.join(src_tensors_folder, x_filename + '.pt')
    if not os.path.exists(sec_tensor_fname):
        output_fn(f'{sec_tensor_fname} does not exist')
    else:
        output_fn(f'Load tensors from {x_filename}.pt')
        X = torch.load(sec_tensor_fname)
        assert X.dim() == 2
        output_fn(f'Loaded tensor X shape {X.shape}')

        output_fn(f'Loading tensor y from {y_filename}')
        y = torch.load(os.path.join(src_tensors_folder, y_filename + '.pt'))
        output_fn(f'Loaded tensor y shape {y.shape} from {y_filename}.pt')

        max_ind = X.shape[0] - (X.shape[0] % WINDOW_SIZE)
        output_fn(f'source {X.shape[0]} window {WINDOW_SIZE} max {max_ind}')
        X_windowed, y_windowed = get_X_y_windowed(X, y, max_ind, output_fn)

        if window_shifts:
            X_windowed_list = [X_windowed]
            y_windowed_list = [y_windowed]
            min_win_num = X_windowed.shape[0]
            for shift in window_shifts:
                assert shift < WINDOW_SIZE
                output_fn(f"Create shifted windows for shift {shift}")
                x_shape_0 = X.shape[0] - shift
                max_ind = x_shape_0 - (x_shape_0 % WINDOW_SIZE)
                X_windowed_shifted, y_windowed_shifted = get_X_y_windowed(X[shift:], y[shift:], max_ind, output_fn)
                X_windowed_list.append(X_windowed_shifted)
                y_windowed_list.append(y_windowed_shifted)
                assert X_windowed.shape[0] == y_windowed.shape[0], 'Window number not equals label number'
                min_win_num = min(min_win_num, X_windowed_shifted.shape[0])

            X_windowed = torch.concat([x_win[:min_win_num, :, :] for x_win in X_windowed_list], dim=1) \
                .reshape(-1, WINDOW_SIZE, X.shape[1])
            y_windowed = torch.concat([y_win[:min_win_num] for y_win in y_windowed_list], dim=1).reshape(-1, WINDOW_SIZE)

        del X, y
        output_fn('Filter out windows with non-unique labels')
        unique_indices = [i for i, e in enumerate([list(set(y_windowed[i].tolist())) for i in range(y_windowed.shape[0])])
                          if (len(e) == 1 and e[0] == round(e[0]))]
        X_windowed = X_windowed[unique_indices]
        y_windowed = y_windowed[unique_indices].mean(dim=1)
        output_fn(f'Left with {len(unique_indices)} windows. X_windowed shape {X_windowed.shape}'
                  f' y_windowed shape {y_windowed.shape}')

        output_fn('Saving X_windowed')
        torch.save(X_windowed, os.path.join(windowed_tensors_folder, x_filename + '_windowed.pt'))
        with open(os.path.join(windowed_tensors_folder, x_filename + '_windowed.npy'), 'wb') as f:
            np.save(f, X_windowed.numpy())
        del X_windowed
        output_fn(f'EMG windows saved to file saved to file {x_filename}_windowed.pt')

        output_fn('Saving y_windowed')
        torch.save(y_windowed, os.path.join(windowed_tensors_folder, y_filename + '_windowed.pt'))
        with open(os.path.join(windowed_tensors_folder, y_filename + '_windowed.npy'), 'wb') as f:
            np.save(f, y_windowed.numpy())
        del y_windowed
        output_fn(f'Labels saved to file {y_filename}_windowed.pt')


def get_X_y_windowed(X, y, max_ind, output_fn):
    X_windowed = X[:max_ind, :].reshape(-1, WINDOW_SIZE, X.shape[1])
    output_fn(f'Finished windowing X_windowed shape {X_windowed.shape}')
    y_windowed = y[:max_ind].reshape(-1, WINDOW_SIZE)
    output_fn(f'Finished windowing y_windowed shape {y_windowed.shape}')
    return X_windowed, y_windowed


if __name__ == '__main__':
    main()
