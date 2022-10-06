import logging
import os

import torch
import wandb
from common import utils
from federated_private_emg.fed_priv_models.model3d import Model3d
from common.utils import train_model, init_data_loaders
from common.config import Config


def main():
    exp_name = utils.get_exp_name(os.path.basename(__file__)[:-3])
    logger = utils.config_logger(f'{exp_name}_logger',
                                 level=logging.INFO, log_folder='../log/')
    logger.info(exp_name)
    if Config.WRITE_TO_WANDB:
        wandb.init(project="emg_gp_moshe", entity="emg_diff_priv", name=exp_name)
        # wandb.config.update({})

    test_loader, train_loader = init_data_loaders(datasets_folder_name=Config.WINDOWED_DATA_DIR, output_fn=logger.info)

    model = Model3d(number_of_classes=Config.NUM_CLASSES, window_size=Config.WINDOW_SIZE, output_info_fn=logger.info,
                    output_debug_fn=logger.debug)
    optimizer = torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY,
                                momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    train_model(criterion, model,
                optimizer,
                train_loader,
                test_loader,
                eval_every=Config.EVAL_EVERY,
                num_epochs=Config.NUM_EPOCHS)


if __name__ == '__main__':
    main()
