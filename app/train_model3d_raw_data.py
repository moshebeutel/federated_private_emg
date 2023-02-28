import logging
import os

import torch
import wandb
from differential_privacy.params import DpParams
from federated_private_emg.fed_priv_models.model3d import Model3d
from common.utils import init_data_loaders, get_exp_name, config_logger
from train.params import TrainParams
from train.train_objects import TrainObjects
from train.train_utils import train_model
from common.config import Config


def main():
    exp_name = get_exp_name(os.path.basename(__file__)[:-3])
    logger = config_logger(f'{exp_name}_logger',
                           level=logging.INFO, log_folder='../log/')
    logger.info(exp_name)
    if Config.WRITE_TO_WANDB:
        wandb.init(project="emg_gp_moshe", entity="emg_diff_priv", name=exp_name)
        # wandb.config.update({})

    train_loader, validation_loader, test_loader = init_data_loaders(datasets_folder_name=Config.WINDOWED_DATA_DIR,
                                                                     batch_size=Config.BATCH_SIZE,
                                                                     output_fn=logger.info)

    model = Model3d(number_of_classes=Config.NUM_CLASSES,
                    window_size=Config.WINDOW_SIZE,
                    use_dropout=Config.USE_DROPOUT,
                    use_group_norm=Config.USE_BATCHNORM,
                    output_info_fn=logger.info,
                    output_debug_fn=logger.debug)
    model.to(Config.DEVICE)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=Config.LEARNING_RATE,
                                weight_decay=Config.WEIGHT_DECAY,
                                momentum=Config.MOMENTUM)
    train_objects = TrainObjects(model=model, loader=train_loader, optimizer=optimizer)
    train_params = TrainParams(epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE,
                               descent_every=Config.LOT_SIZE_IN_BATCHES, validation_every=Config.EVAL_EVERY,
                               test_at_end=Config.TEST_AT_END, add_dp_noise=Config.ADD_DP_NOISE)
    dp_params = DpParams(dp_lot=Config.LOT_SIZE_IN_BATCHES * Config.BATCH_SIZE,
                         dp_sigma=Config.DP_SIGMA, dp_C=Config.DP_C)
    train_model(train_objects=train_objects,
                validation_loader=validation_loader,
                test_loader=test_loader,
                train_params=train_params,
                dp_params=dp_params,
                log2wandb=Config.WRITE_TO_WANDB,
                output_fn=logger.info)


if __name__ == '__main__':
    main()
