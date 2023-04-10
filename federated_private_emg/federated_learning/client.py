import copy
import os

import torch

from common import utils
from common.config import Config
from common.utils import init_data_loaders


class Client:
    def __init__(self, name, model):
        self._user_data = None
        self._user_labels = None
        self._username = name
        self._user_dataset_folder_name = os.path.join(Config.WINDOWED_DATA_DIR, name)
        self._test_loader, self._validation_loader, self._train_loader = \
            init_data_loaders(datasets_folder_name=self._user_dataset_folder_name)
        device = next(model.parameters()).device
        self._device = device
        self._local_model = copy.deepcopy(model).to(device)
        self._local_optimizer = torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE,
                                                weight_decay=Config.WEIGHT_DECAY, momentum=Config.MOMENTUM)

    @property
    def username(self):
        return self._username

    @username.setter
    def username(self, val):
        self._username = val

    def load_dataset(self, exclude_labels=[]):
        assert self._user_data is None, f'User {self.username} data already loaded'
        assert self._user_labels is None, f'User {self.username} labels already loaded'
        dataset = 'train'
        self._user_data, self._user_labels = utils.load_datasets(datasets_folder_name=self._user_dataset_folder_name,
                                                                 x_filename=f'X_{dataset}_windowed.pt',
                                                                 y_filename=f'y_{dataset}_windowed.pt',
                                                                 exclude_labels=exclude_labels)

    def train_single_epoch(self, checkpoint: dict):
        assert isinstance(checkpoint, dict), f'Expected dict checkpoint. Got {type(checkpoint)}'
        assert 'model_state_dict' in checkpoint, 'Expected model_state_dict in checkpoint'
        assert 'optimizer_state_dict' in checkpoint, 'Expected optimizer_state_dict in checkpoint'

        self._local_model.load_state_dict(checkpoint['model_state_dict'])
        self._local_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])





