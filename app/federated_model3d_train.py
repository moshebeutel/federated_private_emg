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
    # logger = utils.config_logger(f'{exp_name}_logger',
    #                              level=logging.INFO, log_folder='../log/')
    # logger.info(exp_name)
    if Config.WRITE_TO_WANDB:
        wandb.init(project="emg_gp_moshe", entity="emg_diff_priv", name=exp_name)
        wandb.config.update({'batch_size': Config.BATCH_SIZE, 'learning_rate': Config.LEARNING_RATE})

    single_train()


def single_train():
    exp_name = utils.get_exp_name(os.path.basename(__file__)[:-3])
    logger = utils.config_logger(f'{exp_name}_logger',
                                 level=logging.INFO, log_folder='../log/')
    logger.info(exp_name)
    model = Model3d(number_of_classes=Config.NUM_CLASSES,
                    window_size=Config.WINDOW_SIZE,
                    use_group_norm=True,
                    output_info_fn=logger.info,
                    output_debug_fn=logger.debug)
    model.to(Config.DEVICE)
    dp_params = DpParams(dp_lot=Config.LOT_SIZE_IN_BATCHES * Config.BATCH_SIZE, dp_sigma=Config.DP_SIGMA,
                         dp_C=Config.DP_C)
    internal_train_params = TrainParams(epochs=Config.NUM_INTERNAL_EPOCHS, batch_size=Config.BATCH_SIZE,
                                        descent_every=Config.LOT_SIZE_IN_BATCHES, validation_every=Config.EVAL_EVERY,
                                        test_at_end=False)
    federated_train_model(model=model,
                          train_user_list=['03', '05', '06', '08', '09', '11', '13', '14', '15', '16',
                                           '17', '18', '19', '24', '25', '26', '27', '29', '30', '31',
                                           '33', '34', '35', '36', '38', '39', '42', '43', '45', '46'],

                          validation_user_list=['22', '23', '47'],
                          test_user_list=['07', '12', '48'],
                          internal_train_params=internal_train_params,
                          num_epochs=Config.NUM_EPOCHS,
                          dp_params=dp_params if Config.ADD_DP_NOISE else None,
                          log2wandb=Config.WRITE_TO_WANDB,
                          output_fn=logger.info)


def sweep_train(config=None):
    exp_name = utils.get_exp_name(os.path.basename(__file__)[:-3])
    with wandb.init(config=config):
        config = wandb.config

        config.LEARNING_RATE = config.learning_rate
        config.BATCH_SIZE = config.batch_size
        config.EPOCHS = 50

        single_train()


def run_sweep():
    sweep_config = {
        'method': 'grid'
    }
    parameters_dict = {}

    sweep_config['parameters'] = parameters_dict
    metric = {
        'name': 'epoch_validation_acc',
        'goal': 'maximize'
    }

    sweep_config['metric'] = metric

    # parameters_dict.update({
    #     'epochs': {
    #         'value': 50},
    #     })

    parameters_dict.update({
        'learning_rate': {
            'values': [0.00001, 0.0001, 0.001, 0.01, 0.1]
        },
        'batch_size': {
            'values': [128, 256, 512]
        }
    })

    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

    wandb.agent(sweep_id, sweep_train)


if __name__ == '__main__':
    main()
    # run_sweep()
