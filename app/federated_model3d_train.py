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
    sigma = f"{'%.3f' % Config.GEP_SIGMA0},{'%.3f' % Config.GEP_SIGMA1}" if Config.USE_GEP \
        else f"{'%.3f' % Config.DP_SIGMA}"
    clip = f"{'%.3f' % Config.GEP_CLIP0},{'%.3f' % Config.GEP_CLIP1}" if Config.USE_GEP else f"{'%.3f' % Config.DP_C}"
    exp_name = f'{Config.DP_METHOD},clip={clip},sigma={sigma},num_bases={Config.NUM_CLIENTS_PUBLIC},seed={Config.SEED}'
    sanity = 'SANITY' if Config.SANITY_CHECK else ''
    print(f'exp_name:{sanity}', exp_name)

    return exp_name


def main():
    if Config.TOY_STORY:
        print('USERS_BIASES', {u: '%.3f' % b for u, b in USERS_BIASES.items()})

        print('USERS_VARIANCES', {u: '%.3f' % b for u, b in USERS_VARIANCES.items()})
    # update_accountant_params()
    Config.GEP_USE_PCA = 1
    sigma = 12.79 # 107.46
    if Config.USE_GEP:
        Config.GEP_SIGMA0 = sigma
        Config.GEP_SIGMA1 = sigma
    else:
        Config.DP_SIGMA = sigma
    clip = 0.01
    if Config.USE_GEP:
        Config.GEP_CLIP0 = clip
        Config.GEP_CLIP1 = clip / 5.0
    else:
        Config.DP_C = clip

    Config.SEED = 20
    # Config.NUM_CLIENTS_PUBLIC = 150

    exp_name = get_exp_name()

    if Config.WRITE_TO_WANDB:
        wandb.init(project="emg_gp_moshe", entity="emg_diff_priv", name=exp_name)
        config_dict = Config.to_dict()

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

        # print('Public Class Partitions')
        utils.CLASSES_OF_PUBLIC_USERS = [utils.CIFAR_USER_CLS_PARTITIONS[u][0][0] for u in public_users]
        # print(utils.CLASSES_OF_PUBLIC_USERS)
        # for u in public_users:
        #     print(u, ':', utils.CIFAR_USER_CLS_PARTITIONS[u])

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
    sampling_prob = Config.NUM_CLIENT_AGG / Config.NUM_CLIENTS_PRIVATE
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

    # output_fn(accountant_params_string())


def sweep_train(sweep_id, config=None):
    with wandb.init(config=config):

        config = wandb.config
        config.update({'sweep_id': sweep_id})
        print(config)

        # if config.dp not in ['GEP_RESIDUALS', 'GEP_NO_RESIDUALS'] and not (config.gep_num_bases == 30
        #                                                                    and config.gep_power_iter == 1):
        #     return

        # if config.dp == 'NO_DP' and not (config.epsilon == 8.0 and config.gep_num_bases == 30
        #                                  and config.gep_power_iter == 1):
        #     return

        # if config.dp == 'NO_DP' and not config.epsilon == 8.0:
        #     return

        Config.SANITY_CHECK = True

        Config.USE_GP = (config.use_gp == 1)
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

        Config.CLASSES_PER_USER = config.classes_each_user

        Config.NUM_CLIENTS_PUBLIC = config.num_clients_public
        Config.GEP_HISTORY_GRADS = config.num_clients_public
        Config.GEP_NUM_BASES = config.num_clients_public
        Config.GEP_NUM_GROUPS = 1
        Config.GEP_USE_PCA = 1
        Config.ADD_DP_NOISE = True
        Config.NUM_CLIENT_AGG = config.agg

        if Config.USE_GEP:
            Config.GEP_CLIP0 = config.clip
            Config.GEP_CLIP1 = config.clip / 5.0
        else:
            Config.DP_C = config.clip

        # if Config.DP_METHOD != Config.DP_METHOD_TYPE.NO_DP:
        #     Config.DP_EPSILON = config.epsilon

        sigma = config.sigma
        if Config.USE_GEP:
            Config.GEP_SIGMA0 = sigma
            Config.GEP_SIGMA1 = sigma
        else:
            Config.DP_SIGMA = sigma

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

    parameters_dict.update({
        'clip': {
            'values': [0.01]
        },
        'sigma': {
            'values': [107.46, 12.79, 4.722, 2.016, 0.0]
            # ϵ = 0.01→ noise - multiplier = 874.16
            # ϵ = 0.1→ noise - multiplier = 107.46
            # ϵ = 1.0→ noise - multipllier = 12.79
            # ϵ = 3.0→ noise - multipllier = 4.722
            # ϵ = 8.0→ noise - multipllier = 2.016
        },
        'seed': {
            'values': [20]
            # 'values': [20, 40, 60]
        },
        # 'sample_with_replacement': {
        #     'values': [0, 1]
        # },
        'agg': {
            'values': [50]
        },
        'dp': {
            'values': ['GEP_NO_RESIDUALS', 'GEP_RESIDUALS', 'SGD_DP']
            # 'values': ['GEP_NO_RESIDUALS']
        },
        'num_clients_public': {
            'values': [50]
        },
        'classes_each_user': {
            # 'values': [2, 6, 10]
            'values': [2]
        },
        'use_gp': {
            'values': [0]
            # 'values': [0, 1]
        },
        # 'gep_num_bases': {
        #     'values': [150]
        # },
    })

    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

    wandb.agent(sweep_id, partial(sweep_train, sweep_id=sweep_id))


if __name__ == '__main__':
    # main()
    run_sweep()
