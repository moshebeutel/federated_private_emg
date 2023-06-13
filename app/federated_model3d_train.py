import logging

import torch.nn

import wandb
from common import utils
from common.config import Config
from common.utils import USERS_BIASES, USERS_VARIANCES, public_users, train_user_list, validation_user_list, \
    test_user_list, gen_random_loaders
from differential_privacy.accountant_utils import get_sigma, accountant_params_string
from differential_privacy.params import DpParams
from differential_privacy.utils import attach_gep
from fed_priv_models.model_factory import init_model
from fed_priv_models.pFedGP.Learner import pFedGPFullLearner
from fed_priv_models.pFedGP.utils import set_seed
from train.federated_utils import federated_train_model
from train.params import TrainParams


def get_exp_name_():
    sigma = f'[{"%.3f" % Config.GEP_SIGMA0},{"%.3f" % Config.GEP_SIGMA1}' if Config.USE_GEP else Config.DP_SIGMA
    # sigma0 = '%.3f' % Config.GEP_SIGMA0
    # sigma1 = '%.3f' % Config.GEP_SIGMA1
    # clip0 = '%.3f' % Config.GEP_CLIP0
    # clip1 = '%.3f' % Config.GEP_CLIP1
    clip = f'[{"%.3f" % Config.GEP_CLIP0},{"%.3f" % Config.GEP_CLIP1}' if Config.USE_GEP else Config.DP_C
    # exp_name = f'CIFAR10 GEP eps={Config.DP_EPSILON} delta={Config.DP_DELTA} q={q} sigma=[{sigma0},{sigma1}] clip=[{clip0},{clip1}]'
    # exp_name = f'CIFAR10 GEP clip=[{clip0},{clip1}] eps={Config.DP_EPSILON} delta={Config.DP_DELTA} ' \
    #            f'sigma=[{sigma0},{sigma1}] public users {len(utils.public_users)}' \
    #            f' residual gradients {Config.GEP_USE_RESIDUAL}'
    use_residual = 'using' if Config.GEP_USE_RESIDUAL else 'not using'
    use_residual = (use_residual + ' residual grads') if Config.USE_GEP else ''
    exp_name = f'{Config.DATASET.name} {Config.DP_METHOD.name} clip={clip} sigma={sigma} {use_residual}'
    return exp_name


def main():
    if Config.TOY_STORY:
        print('USERS_BIASES', {u: '%.3f' % b for u, b in USERS_BIASES.items()})

        print('USERS_VARIANCES', {u: '%.3f' % b for u, b in USERS_VARIANCES.items()})

    q = '%.3f' % (float(Config.NUM_CLIENT_AGG) / float(len(train_user_list)))

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

    exp_name = get_exp_name_()

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

    single_train(exp_name)


def single_train(exp_name):
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
        for u in train_user_list:
            GPs[u] = pFedGPFullLearner(n_output=Config.CIFAR10_CLASSES_PER_USER)

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
        public_inputs, public_targets = None, None  # create_public_dataset(public_users=public_users)

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
                          GPs=GPs if Config.USE_GP else None,
                          log2wandb=Config.WRITE_TO_WANDB,
                          output_fn=logger.info)


def update_accountant_params(output_fn):
    sampling_prob = Config.NUM_CLIENT_AGG / (Config.NUM_CLIENTS_TRAIN - Config.NUM_CLIENTS_PUBLIC)
    steps = Config.NUM_EPOCHS  # int(Config.NUM_EPOCHS / sampling_prob)
    output_fn(f'steps {steps}')
    output_fn(f'sampling prob {sampling_prob}')

    if Config.DP_METHOD == Config.DP_METHOD_TYPE.NO_DP:
        sigma = 0.0
    else:
        sigma, eps = get_sigma(sampling_prob, steps, Config.DP_EPSILON,
                               Config.DP_DELTA, rgp=Config.USE_GEP and Config.GEP_USE_RESIDUAL)

    if Config.USE_GEP:
        # sigma_, eps_ = get_sigma(sampling_prob, steps, Config.DP_EPSILON, Config.DP_DELTA, rgp=(not Config.GEP_USE_RESIDUAL))
        # print(f'Epsilon {eps}, sigma {sigma} for use residual {Config.GEP_USE_RESIDUAL}')
        # print(f'Epsilon {eps_}, sigma {sigma_} for use residual {not Config.GEP_USE_RESIDUAL}')

        Config.GEP_SIGMA0 = sigma
        Config.GEP_SIGMA1 = sigma
    else:
        Config.DP_SIGMA = sigma

    output_fn(accountant_params_string())


def sweep_train(config=None):
    update_accountant_params()

    exp_name = get_exp_name_()

    print(exp_name)

    with wandb.init(config=config):
        # config = {**{'_NAME': exp_name}, **wandb.config}
        config = wandb.config
        config['clip'] = 0.01
        print(config)
        Config.EPOCHS = 50

        Config.DP_METHOD = Config.DP_METHOD_TYPE.SGD_DP if config.dp == 'SGD_DP' else Config.DP_METHOD_TYPE.GEP
        Config.USE_GEP = (Config.DP_METHOD == Config.DP_METHOD_TYPE.GEP)
        Config.USE_SGD_DP = (Config.DP_METHOD == Config.DP_METHOD_TYPE.SGD_DP)
        Config.GEP_USE_RESIDUAL = (Config.USE_GEP and config.dp == 'GEP_RESIDUALS')

        if Config.USE_GEP:
            Config.GEP_CLIP0 = config.clip
            Config.GEP_CLIP1 = config.clip / 5.0
        else:
            Config.DP_C = config.clip
        # Config.GEP_SIGMA0 = config.sigma
        # Config.GEP_SIGMA1 = config.sigma

        Config.DP_EPSILON = config.epsilon
        Config.NUM_CLIENT_AGG = 50 - Config.NUM_CLIENTS_PUBLIC
        Config.SAMPLE_CLIENTS_WITH_REPLACEMENT = (config.sample_with_replacement == 1)

        # sigma0 = '%.3f' % Config.GEP_SIGMA0
        # sigma1 = '%.3f' % Config.GEP_SIGMA1
        # clip0 = '%.3f' % Config.GEP_CLIP0
        # clip1 = '%.3f' % Config.GEP_CLIP1

        # exp_name = f'CIFAR10 GEP clip=[{clip0},{clip1} sigma=[{sigma0},{sigma1}] residual gradients {Config.GEP_USE_RESIDUAL}'
        # config.LEARNING_RATE = config.learning_rate
        # config.BATCH_SIZE = config.batch_size
        config['_NAME'] = exp_name
        config.update(Config.to_dict())
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
        # 'clip': {
        #     'values': [0.1, 0.01]
        # },
        # 'sigma': {
        #     'values': [1.2, 3.2, 9.6, 0.6, 1.6, 4.8]
        # },
        'epsilon': {
            'values': [8.0, 3.0, 1.0]
        },
        'sample_with_replacement': {
            'values': [0, 1]
        },
        # 'agg': {
        #     'values': [50]
        # },
        'dp': {
            'values': ['GEP_NO_RESIDUALS', 'GEP_RESIDUALS', 'SGD_DP', ]
        },
        # 'classes_each_user': {
        #     'values': [3]
        # },
        'internal_epochs': {
            'values': [1, 3, 5]
        }

    })

    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

    wandb.agent(sweep_id, sweep_train)


if __name__ == '__main__':
    main()
    # run_sweep()
