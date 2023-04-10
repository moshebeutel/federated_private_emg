import logging
import os

import torch.nn

import wandb
from common import utils
from common.utils import USERS_BIASES, USERS_VARIANCES
from differential_privacy.params import DpParams
from fed_priv_models.pad_operators import PadLastDimCircular, Reshape3Bands, FlattenToLinear, PadBeforeLast, Squeeze
from federated_private_emg.fed_priv_models.model3d import Model3d
from train.federated_utils import federated_train_model, attach_gep, create_public_dataset
from train.params import TrainParams
from common.config import Config
from functools import *

# public_users = ['04', '13', '35', '08']
train_user_list = ['03', '05', '06', '09', '11', '14', '16',
                   '17', '18', '19', '25', '26', '27', '29',
                   '33', '34', '36', '38',
                   '04', '13', '35', '08', '15', '24', '30', '31', '39',
                   '42', '43', '45', '46']
# train_user_list=['04', '13', '35']
# train_user_list=['04', '13', '35', '08', '17', '18', '19', '25', '26', '27', '29']
# train_user_list=['04']
validation_user_list = ['22', '23', '47']
# validation_user_list=['04']
test_user_list = ['07', '12', '48']


def main():
    q = '%.3f' % (float(Config.NUM_CLIENT_AGG) / float(len(train_user_list)))
    sigma0 = '%.3f' % Config.GEP_SIGMA0
    sigma1 = '%.3f' % Config.GEP_SIGMA1
    clip0 = '%.3f' % Config.GEP_CLIP0
    clip1 = '%.3f' % Config.GEP_CLIP1

    dp_sigma = '%.3f' % Config.DP_SIGMA
    dp_c = '%.3f' % Config.DP_C

    if Config.USE_GEP:
        exp_name = f'High Dim GEP eps={Config.DP_EPSILON} delta={Config.DP_DELTA} q={q} sigma=[{sigma0},{sigma1}] clip=[{clip0},{clip1}]'
    elif Config.USE_SGD_DP:
        exp_name = f'High Dim SGD_DP eps={Config.DP_EPSILON} delta={Config.DP_DELTA} q={q} sigma={dp_sigma} C={dp_c}'
    # exp_name = utils.get_exp_name('TOY STORY Federated')
    # logger = utils.config_logger(f'{exp_name}_logger',
    #                              level=logging.INFO, log_folder='../log/')
    # logger.info(exp_name)
    print(exp_name)
    if Config.WRITE_TO_WANDB:
        wandb.init(project="emg_gp_moshe", entity="emg_diff_priv", name=exp_name)
        config_dict = Config.to_dict()
        users = train_user_list + validation_user_list + test_user_list
        for u in users:
            config_dict[f'user_{u}_bias'] = USERS_BIASES[u]
        for u in users:
            config_dict[f'user_{u}_variance'] = USERS_VARIANCES[u]
        config_dict.update({'train_user_list': train_user_list,
                            'validation_user_list': validation_user_list,
                            'test_user_list': test_user_list})
        wandb.config.update(config_dict)

    single_train()


def single_train():
    exp_name = utils.get_exp_name(os.path.basename(__file__)[:-3])
    logger = utils.config_logger(f'{exp_name}_logger',
                                 level=logging.INFO, log_folder='../log/')
    logger.info(exp_name)

    # model = Model3d(number_of_classes=Config.NUM_CLASSES,
    #                 window_size=Config.WINDOW_SIZE,
    #                 use_group_norm=True,
    #                 output_info_fn=logger.info,
    #                 output_debug_fn=logger.debug)

    if Config.TOY_STORY:
        model = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            # DenseBlock
            torch.nn.Linear(in_features=Config.DATA_DIM, out_features=Config.HIDDEN_DIM, bias=True),
            # torch.nn.Linear(in_features=1000, out_features=100, bias=True),
            torch.nn.Linear(in_features=Config.HIDDEN_DIM, out_features=Config.OUTPUT_DIM, bias=True))
    else:
        model = torch.nn.Sequential(torch.nn.Sequential(

            Reshape3Bands(window_size=Config.WINDOW_SIZE, W=3, H=8),
            PadBeforeLast(),
            PadLastDimCircular(window_size=Config.WINDOW_SIZE, W=3),

            # Conv3DBlock
            torch.nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            torch.nn.AvgPool3d(kernel_size=(1, 3, 1), stride=(1, 3, 1), padding=0),
            Squeeze(),
            # torch.nn.GroupNorm(4, 32, eps=1e-05, affine=True),
            torch.nn.ReLU(inplace=False),

            # Conv2DBlock
            torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2)),
            torch.nn.AvgPool2d(kernel_size=(3, 3), stride=(3, 3), padding=0),
            Squeeze(),
            # torch.nn.GroupNorm(4, 64, eps=1e-05, affine=True),
            torch.nn.ReLU(inplace=False),

            # Conv1DBlock
            torch.nn.Conv1d(64, 128, kernel_size=(3,), stride=(2,)),
            torch.nn.AvgPool1d(kernel_size=(3,), stride=(3,), padding=(0,)),
            # torch.nn.GroupNorm(4, 128, eps=1e-05, affine=True),
            torch.nn.ReLU(inplace=False),

            # Conv1DBlock
            torch.nn.Conv1d(128, 256, kernel_size=(3,), stride=(2,)),
            torch.nn.AvgPool1d(kernel_size=(3,), stride=(3,), padding=(0,)),
            # torch.nn.GroupNorm(4, 128, eps=1e-05, affine=True),
            torch.nn.ReLU(inplace=False),

            # FlattenToLinear(),
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            # DenseBlock
            torch.nn.Linear(in_features=256, out_features=256, bias=True),
            torch.nn.ReLU(inplace=False),

            # DenseBlock
            torch.nn.Linear(in_features=256, out_features=128, bias=True),
            torch.nn.ReLU(inplace=False),

            # DenseBlock
            torch.nn.Linear(in_features=128, out_features=64, bias=True),
            torch.nn.ReLU(inplace=False),

            # output layer
            torch.nn.Linear(in_features=64, out_features=7, bias=True)
        ))

    model.to(Config.DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss() if not Config.TOY_STORY else torch.nn.MSELoss()
    if Config.USE_GEP:
        # public_users = ['04', '13', '35', '08', '15', '24', '30', '31', '39', '42', '43', '45', '46']
        public_users = ['04', '13', '35', '08']
        # public_users = ['04']
        public_inputs, public_targets = create_public_dataset(public_users=public_users)

        attach_gep_to_model = partial(attach_gep, loss_fn=loss_fn, num_bases=Config.GEP_NUM_BASES,
                                      batch_size=Config.BATCH_SIZE if Config.INTERNAL_BENCHMARK else Config.NUM_CLIENT_AGG,
                                      clip0=Config.GEP_CLIP0, clip1=Config.GEP_CLIP1,
                                      power_iter=Config.GEP_POWER_ITER, num_groups=Config.GEP_NUM_GROUPS,
                                      public_inputs=public_inputs, public_targets=public_targets)

    dp_params = DpParams(dp_lot=Config.LOT_SIZE_IN_BATCHES * Config.BATCH_SIZE, dp_sigma=Config.DP_SIGMA,
                         dp_C=Config.DP_C)

    internal_train_params = TrainParams(epochs=Config.NUM_INTERNAL_EPOCHS,
                                        batch_size=Config.BATCH_SIZE,
                                        descent_every=Config.LOT_SIZE_IN_BATCHES,
                                        validation_every=Config.EVAL_EVERY,
                                        test_at_end=False)

    federated_train_model(model=model, loss_fn=loss_fn,
                          train_user_list=train_user_list,
                          validation_user_list=validation_user_list,
                          test_user_list=test_user_list,
                          internal_train_params=internal_train_params,
                          num_epochs=Config.NUM_EPOCHS,
                          dp_params=dp_params if Config.ADD_DP_NOISE else None,
                          attach_gep_to_model_fn=None if not Config.USE_GEP else attach_gep_to_model,
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
