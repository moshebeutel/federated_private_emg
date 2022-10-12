import logging
import os

import wandb
from common import utils
from differential_privacy.params import DpParams
from federated_private_emg.fed_priv_models.model3d import Model3d
from train.federated_utils import federated_train_model
from train.params import TrainParams
from common.config import Config


def main():
    exp_name = utils.get_exp_name(os.path.basename(__file__)[:-3])
    logger = utils.config_logger(f'{exp_name}_logger',
                                 level=logging.INFO, log_folder='../log/')
    logger.info(exp_name)
    if Config.WRITE_TO_WANDB:
        wandb.init(project="emg_gp_moshe", entity="emg_diff_priv", name=exp_name)
        # wandb.config.update({})

    model = Model3d(number_of_classes=Config.NUM_CLASSES, window_size=Config.WINDOW_SIZE, output_info_fn=logger.info,
                    output_debug_fn=logger.debug)
    model.to(Config.DEVICE)
    dp_params = DpParams(dp_lot=Config.LOT_SIZE_IN_BATCHES * Config.BATCH_SIZE, dp_sigma=Config.DP_SIGMA,
                         dp_C=Config.DP_C)
    internal_train_params = TrainParams(epochs=Config.NUM_INTERNAL_EPOCHS, batch_size=Config.BATCH_SIZE,
                                        descent_every=Config.LOT_SIZE_IN_BATCHES, validation_every=Config.EVAL_EVERY,
                                        test_at_end=False, add_dp_noise=Config.ADD_DP_NOISE)
    federated_train_model(model=model,
                          train_user_list=['03', '04', '05', '08', '09'],
                          validation_user_list=['06'],
                          test_user_list=['07'],
                          internal_train_params=internal_train_params,
                          num_epochs=Config.NUM_EPOCHS,
                          dp_params=dp_params,
                          log2wandb=Config.WRITE_TO_WANDB,
                          output_fn=logger.info)


if __name__ == '__main__':
    main()
