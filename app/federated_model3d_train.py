import logging
from functools import partial

import torch.nn

import wandb
from common import utils
from common.config import Config
from common.utils import USERS_BIASES, USERS_VARIANCES, public_users, train_user_list, validation_user_list, \
    test_user_list, gen_random_loaders
from differential_privacy.accountant_utils import get_sigma, accountant_params_string
from differential_privacy.utils import attach_gep
from fed_priv_models.model_factory import init_model
from fed_priv_models.pFedGP.Learner import pFedGPFullLearner
from fed_priv_models.pFedGP.utils import set_seed
from train.federated_utils import federated_train_model
from train.params import TrainParams


def get_exp_name():
    sigma = f"{'%.5f' % Config.GEP_SIGMA0},{'%.3f' % Config.GEP_SIGMA1}" if Config.USE_GEP \
        else f"{'%.5f' % Config.DP_SIGMA}"
    clip = f"{'%.5f' % Config.GEP_CLIP0},{'%.3f' % Config.GEP_CLIP1}" if Config.USE_GEP else f"{'%.3f' % Config.DP_C}"
    dp = Config.DP_METHOD.name
    dp += f' clip {clip}'
    if dp == 'GEP':
        dp += (' with residuals' if Config.GEP_USE_RESIDUAL else ' without residuals')
        # dp += f' num_bases {Config.GEP_NUM_BASES} power iter {Config.GEP_POWER_ITER}'
    exp_name = f'{Config.DATASET.name} {dp}'
    if dp != 'NO_DP' and Config.ADD_DP_NOISE:
        exp_name += f' epsilon={Config.DP_EPSILON} sigma={sigma}'

    print('exp_name:', exp_name)

    return exp_name


def main():
    if Config.TOY_STORY:
        print('USERS_BIASES', {u: '%.3f' % b for u, b in USERS_BIASES.items()})

        print('USERS_VARIANCES', {u: '%.3f' % b for u, b in USERS_VARIANCES.items()})

    # q = '%.3f' % (float(Config.NUM_CLIENT_AGG) / float(len(train_user_list)))

    # if Config.USE_GEP:
    #     sigma0 = '%.3f' % Config.GEP_SIGMA0
    #     sigma1 = '%.3f' % Config.GEP_SIGMA1
    #     clip0 = '%.3f' % Config.GEP_CLIP0
    #     clip1 = '%.3f' % Config.GEP_CLIP1
    #     # exp_name = f'CIFAR10 GEP eps={Config.DP_EPSILON} delta={Config.DP_DELTA} q={q} sigma=[{sigma0},{sigma1}] clip=[{clip0},{clip1}]'
    #     exp_name = f'CIFAR10 GEP clip=[{clip0},{clip1}] eps={Config.DP_EPSILON} delta={Config.DP_DELTA} q={q} ' \
    #                f'sigma=[{sigma0},{sigma1}] public users {len(utils.public_users)}' \
    #                f' residual gradients {Config.GEP_USE_RESIDUAL}'
    #     # exp_name = f'CIFAR10 GEP clip=[{clip0},{clip1}] no noise 50 train users {len(utils.public_users)} public residual gradients {Config.GEP_USE_RESIDUAL} AGG {Config.NUM_CLIENT_AGG} {Config.CIFAR10_CLASSES_PER_USER} classes each user'
    #     # exp_name = f'High Dim GEP data scale={Config.DATA_SCALE} eps={Config.DP_EPSILON} delta={Config.DP_DELTA} q={q} sigma=[{sigma0},{sigma1}] clip=[{clip0},{clip1}]'
    # elif Config.USE_SGD_DP:
    #     dp_sigma = '%.3f' % Config.DP_SIGMA
    #     dp_c = '%.3f' % Config.DP_C
    #     exp_name = f'EMG SGD_DP eps={Config.DP_EPSILON} delta={Config.DP_DELTA} q={q} sigma={dp_sigma} C={dp_c}'
    #     # exp_name = f'High Dim SGD_DP data scale={Config.DATA_SCALE} eps={Config.DP_EPSILON} delta={Config.DP_DELTA} q={q} sigma={dp_sigma} C={dp_c}'
    # exp_name = utils.get_exp_name('TOY STORY Federated')
    # logger = utils.config_logger(f'{exp_name}_logger',
    #                              level=logging.INFO, log_folder='../log/')
    # logger.info(exp_name)
    # update_accountant_params(output_fn=logging.info)

    update_accountant_params()
    exp_name = get_exp_name()

    if Config.WRITE_TO_WANDB:
        wandb.init(project="emg_gp_moshe", entity="emg_diff_priv", name=exp_name)
        config_dict = Config.to_dict()
        # users = train_user_list + validation_user_list + test_user_list
        # for u in users:
        #     config_dict[f'user_{u}_bias'] = USERS_BIASES[u]
        # for u in users:
        # #     config_dict[f'user_{u}_variance'] = USERS_VARIANCES[u]
        # config_dict.update({'train_user_list': train_user_list,
        #                     'validation_user_list': validation_user_list,
        #                     'test_user_list': test_user_list})

        wandb.config.update(config_dict)

    single_train(exp_name)


def single_train(exp_name):
    if Config.WRITE_TO_WANDB:
        wandb.run.name = exp_name
    set_seed(Config.SEED)
    logger = utils.config_logger(f'{exp_name}_logger',
                                 level=logging.DEBUG, log_folder='../log/')
    logger.info(exp_name)

    model = init_model()

    num_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model has {num_of_parameters} trainable parameters')
    model.to(Config.DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss() if not Config.TOY_STORY else torch.nn.MSELoss()

    if Config.USE_GP:
        GPs = {}
        for u in utils.all_users_list:
            GPs[u] = pFedGPFullLearner(n_output=Config.CLASSES_PER_USER)

    if Config.CIFAR_DATA:
        loaders, cls_partitions = gen_random_loaders(num_users=len(utils.all_users_list), bz=Config.BATCH_SIZE,
                                                     classes_per_user=Config.CLASSES_PER_USER)
        utils.CIFAR_USER_LOADERS = \
            {user: {'train': train_loader, 'validation': validation_loader, 'test': test_loader}
             for user, train_loader, validation_loader, test_loader in
             zip(utils.all_users_list, loaders[0], loaders[1], loaders[2])}

        utils.CIFAR_USER_CLS_PARTITIONS = \
            {user: (cls, prb) for (user, cls, prb) in
             zip(utils.all_users_list, cls_partitions['class'], cls_partitions['prob'])}

        print('Public Class Partitions')
        utils.CLASSES_OF_PUBLIC_USERS = [utils.CIFAR_USER_CLS_PARTITIONS[u][0][0] for u in public_users]
        print(utils.CLASSES_OF_PUBLIC_USERS)
        for u in public_users:
            print(u, ':', utils.CIFAR_USER_CLS_PARTITIONS[u])

    gep = None
    if Config.USE_GEP:
        batch_size_for_gep = Config.BATCH_SIZE if Config.INTERNAL_BENCHMARK else Config.NUM_CLIENT_AGG

        model, loss_fn, gep = attach_gep(net=model,
                                         loss_fn=loss_fn,
                                         num_bases=Config.GEP_NUM_BASES,
                                         batch_size=batch_size_for_gep,
                                         clip0=Config.GEP_CLIP0, clip1=Config.GEP_CLIP1,
                                         power_iter=Config.GEP_POWER_ITER, num_groups=Config.GEP_NUM_GROUPS,
                                         public_users=public_users)

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
                          GPs=GPs if Config.USE_GP else None,
                          log2wandb=Config.WRITE_TO_WANDB,
                          output_fn=logger.info)


def update_accountant_params(output_fn=lambda s: None):
    sampling_prob = Config.NUM_CLIENT_AGG / (Config.NUM_CLIENTS_TRAIN - Config.NUM_CLIENTS_PUBLIC)
    steps = Config.NUM_EPOCHS  # int(Config.NUM_EPOCHS / sampling_prob)
    output_fn(f'steps {steps}')
    output_fn(f'sampling prob {sampling_prob}')

    if Config.DP_METHOD == Config.DP_METHOD_TYPE.NO_DP:
        sigma = 0.0
    else:
        sigma, eps = get_sigma(q=sampling_prob,
                               T=steps,
                               eps=Config.DP_EPSILON,
                               delta=Config.DP_DELTA,
                               rgp=Config.USE_GEP and Config.GEP_USE_RESIDUAL)

    if Config.USE_GEP:
        Config.GEP_SIGMA0 = sigma
        Config.GEP_SIGMA1 = sigma
    else:
        Config.DP_SIGMA = sigma

    output_fn(accountant_params_string())


def sweep_train(sweep_id, config=None):
    with wandb.init(config=config):

        config = wandb.config
        config.update({'sweep_id': sweep_id})

        # if config.dp not in ['GEP_RESIDUALS', 'GEP_NO_RESIDUALS'] and not (config.gep_num_bases == 30
        #                                                                    and config.gep_power_iter == 1):
        #     return

        # if config.dp == 'NO_DP' and not (config.epsilon == 8.0 and config.gep_num_bases == 30
        #                                  and config.gep_power_iter == 1):
        #     return

        # if config.dp == 'NO_DP' and not config.epsilon == 8.0:
        #     return

        # Config.USE_GP = (config.use_gp == 1)
        Config.SEED = config.seed
        if config.dp == 'SGD_DP':
            Config.DP_METHOD = Config.DP_METHOD_TYPE.SGD_DP
            Config.USE_GEP = False
            Config.USE_SGD_DP = True
            Config.GEP_USE_RESIDUAL = False
            Config.ADD_DP_NOISE = True
        elif config.dp == 'NO_DP':
            Config.DP_METHOD = Config.DP_METHOD_TYPE.NO_DP
            Config.USE_GEP = False
            Config.USE_SGD_DP = False
            Config.GEP_USE_RESIDUAL = False
            Config.ADD_DP_NOISE = False
        elif config.dp == 'GEP_RESIDUALS':
            Config.DP_METHOD = Config.DP_METHOD_TYPE.GEP
            Config.USE_GEP = True
            Config.USE_SGD_DP = False
            Config.GEP_USE_RESIDUAL = True
            Config.ADD_DP_NOISE = True
        elif config.dp == 'GEP_NO_RESIDUALS':
            Config.DP_METHOD = Config.DP_METHOD_TYPE.GEP
            Config.USE_GEP = True
            Config.USE_SGD_DP = False
            Config.GEP_USE_RESIDUAL = False
            Config.ADD_DP_NOISE = True

        Config.NUM_CLIENTS_PUBLIC = config.num_clients_public
        Config.HIDDEN_DIM = config.hidden_dim
        Config.GEP_NUM_GROUPS = 1
        Config.GEP_NUM_BASES = Config.NUM_CLIENTS_PUBLIC * Config.GEP_NUM_GROUPS
        # Config.GEP_POWER_ITER = config.gep_power_iter
        # Config.GEP_USE_PCA = (config.use_pca == 1)
        Config.GEP_USE_PCA = 1
        Config.ADD_DP_NOISE = False
        Config.NUM_CLIENT_AGG = config.agg

        if "clip" not in config:
            config['clip'] = 0.001

        if Config.USE_GEP:
            Config.GEP_CLIP0 = config.clip
            Config.GEP_CLIP1 = config.clip / 5.0
        else:
            Config.DP_C = config.clip

        if Config.DP_METHOD != Config.DP_METHOD_TYPE.NO_DP:
            Config.DP_EPSILON = config.epsilon

        update_accountant_params()

        exp_name = get_exp_name()

        print(exp_name)

        config.update({'app_config_dict': Config.to_dict()})

        print(config)

        single_train(exp_name)


def run_sweep():
    assert Config.WRITE_TO_WANDB, f'sweep run expects wandb. Got Config.WRITE_TO_WANDB={Config.WRITE_TO_WANDB}'

    sweep_config = {
        'method': 'grid'
    }
    parameters_dict = {}

    sweep_config['parameters'] = parameters_dict
    metric = {
        'name': 'best_epoch_validation_acc',
        'goal': 'maximize'
    }

    sweep_config['metric'] = metric

    # parameters_dict.update({
    #     'epochs': {
    #         'value': 50},
    #     })

    parameters_dict.update({
        # 'learning_rate': {
        #     'values': [0.00001, 0.0001, 0.001, 0.01, 0.1]
        # },
        # 'batch_size': {
        #     'values': [128, 256, 512]
        # },
        'clip': {
            # 'values': [0.00001, 0.0001, 0.001]
            'values': [0.00001]
        },
        # 'sigma': {
        #     'values': [1.2, 3.2, 9.6, 0.6, 1.6, 4.8]
        # },
        'seed': {
            'values': [10, 50]
            # 'values': [10,30,50]
        },
        'epsilon': {
            'values': [1.0]
            # 'values': [0.5, 0.1, 0.01, 0.001]
        },
        # 'sample_with_replacement': {
        #     'values': [0, 1]
        # },
        'agg': {
            'values': [10]
        },
        'dp': {
            # 'values': ['GEP_NO_RESIDUALS', 'GEP_RESIDUALS', 'SGD_DP', 'NO_DP']
            # 'values': ['GEP_NO_RESIDUALS', 'GEP_RESIDUALS', 'SGD_DP']
            'values': ['GEP_NO_RESIDUALS']
            # 'values': ['SGD_DP']
            # 'values': ['NO_DP']

        },
        # 'use_pca': {
        #     'values': [1, 0]
        # },
        'num_clients_public': {
            'values': [200]
            # 'values': [25, 50, 70, 100]
        },
        'hidden_dim': {
            'values': [15, 25, 30]
        },
        # 'classes_each_user': {
        #     'values': [3]
        # },
        # 'internal_epochs': {
        #     'values': [1, 5]
        # },
        # 'use_gp': {
        #     'values': [0, 1]
        # },
        # 'gep_num_bases': {
        #     'values': [21, 30]
        # },
        #
        # 'gep_num_groups': {
        #     'values': [15, 20]
        # },
        # 'gep_power_iter': {
        #     'values': [1, 5]
        # }
    })

    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

    wandb.agent(sweep_id, partial(sweep_train, sweep_id=sweep_id))


if __name__ == '__main__':
    main()
    # run_sweep()
