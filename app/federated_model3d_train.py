import logging
import os

import torch.nn

import wandb
from common import utils
from common.config import Config
from common.utils import USERS_BIASES, USERS_VARIANCES, public_users, train_user_list, validation_user_list, \
    test_user_list, gen_random_loaders
from differential_privacy.params import DpParams
from differential_privacy.utils import attach_gep
from fed_priv_models.custom_sequential import init_model
from train.federated_utils import federated_train_model, create_public_dataset
from train.params import TrainParams


def main():
    if Config.TOY_STORY:
        print('USERS_BIASES', {u: '%.3f' % b for u, b in USERS_BIASES.items()})

        print('USERS_VARIANCES', {u: '%.3f' % b for u, b in USERS_VARIANCES.items()})

    q = '%.3f' % (float(Config.NUM_CLIENT_AGG) / float(len(train_user_list)))

    if Config.USE_GEP:
        sigma0 = '%.3f' % Config.GEP_SIGMA0
        sigma1 = '%.3f' % Config.GEP_SIGMA1
        clip0 = '%.3f' % Config.GEP_CLIP0
        clip1 = '%.3f' % Config.GEP_CLIP1
        # exp_name = f'CIFAR10 GEP eps={Config.DP_EPSILON} delta={Config.DP_DELTA} q={q} sigma=[{sigma0},{sigma1}] clip=[{clip0},{clip1}]'
        exp_name = f'CIFAR10 GEP clip=[{clip0},{clip1}] no noise 50 train users {len(utils.public_users)} public residual gradients {Config.GEP_USE_RESIDUAL} AGG {Config.NUM_CLIENT_AGG} {Config.CIFAR10_CLASSES_PER_USER} classes each user'
        # exp_name = f'High Dim GEP data scale={Config.DATA_SCALE} eps={Config.DP_EPSILON} delta={Config.DP_DELTA} q={q} sigma=[{sigma0},{sigma1}] clip=[{clip0},{clip1}]'
    elif Config.USE_SGD_DP:
        dp_sigma = '%.3f' % Config.DP_SIGMA
        dp_c = '%.3f' % Config.DP_C
        exp_name = f'EMG SGD_DP eps={Config.DP_EPSILON} delta={Config.DP_DELTA} q={q} sigma={dp_sigma} C={dp_c}'
        # exp_name = f'High Dim SGD_DP data scale={Config.DATA_SCALE} eps={Config.DP_EPSILON} delta={Config.DP_DELTA} q={q} sigma={dp_sigma} C={dp_c}'
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

    model = init_model()

    num_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model has {num_of_parameters} trainable parameters')
    model.to(Config.DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss() if not Config.TOY_STORY else torch.nn.MSELoss()

    if Config.CIFAR10_DATA:
        loaders, cls_partitions = gen_random_loaders(num_users=len(utils.all_users_list), bz=Config.BATCH_SIZE,
                                                     classes_per_user=Config.CIFAR10_CLASSES_PER_USER)
        utils.CIFAR10_USER_LOADERS = \
            {user: {'train': train_loader, 'validation': validation_loader, 'test': test_loader}
             for user, train_loader, validation_loader, test_loader in
             zip(utils.all_users_list, loaders[0], loaders[1], loaders[2])}

        utils.CIFAR10_USER_CLS_PARTITIONS = \
            {user: (cls, prb) for (user, cls, prb) in
             zip(utils.all_users_list, cls_partitions['class'], cls_partitions['prob'])}

        print('Public Class Partitions')
        utils.CLASSES_OF_PUBLIC_USERS = [utils.CIFAR10_USER_CLS_PARTITIONS[u][0][0] for u in public_users]
        print(utils.CLASSES_OF_PUBLIC_USERS)
        for u in public_users:
            print(u, ':', utils.CIFAR10_USER_CLS_PARTITIONS[u])

    gep = None
    if Config.USE_GEP:
        public_inputs, public_targets = None, None #  create_public_dataset(public_users=public_users)

        batch_size_for_gep = Config.BATCH_SIZE if Config.INTERNAL_BENCHMARK else len(public_users)

        model, loss_fn, gep = attach_gep(net=model,
                                         loss_fn=loss_fn,
                                         num_bases=Config.GEP_NUM_BASES,
                                         batch_size=batch_size_for_gep,
                                         clip0=Config.GEP_CLIP0, clip1=Config.GEP_CLIP1,
                                         power_iter=Config.GEP_POWER_ITER, num_groups=Config.GEP_NUM_GROUPS,
                                         public_inputs=public_inputs, public_targets=public_targets,
                                         public_users=public_users)

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
                          gep=None if not Config.USE_GEP else gep,
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
