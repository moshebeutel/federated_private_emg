import copy
import logging
import os
from collections import OrderedDict

import torch
from tqdm import tqdm

import wandb
from common import utils
from differential_privacy.params import DpParams
from federated_private_emg.fed_priv_models.model3d import Model3d
from common.utils import init_data_loaders
from train.params import TrainParams
from train.train_objects import TrainObjects
from train.train_utils import run_single_epoch, train_model
from common.config import Config


def federated_train_single_epoch(model, train_user_list,
                                 train_params: TrainParams,
                                 dp_params=None,
                                 output_fn=lambda s: None):
    params = OrderedDict()
    for n, p in model.named_parameters():
        params[n] = torch.zeros_like(p.data)
    epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc = 0.0, 0.0, 0.0, 0.0
    # user_pbar = tqdm(range(len(train_user_list)))
    # for i in user_pbar:
    for u in train_user_list:
        # u = train_user_list[i]
        user_dataset_folder_name = os.path.join(Config.WINDOWED_DATA_DIR, u)
        test_loader, validation_loader, train_loader = init_data_loaders(datasets_folder_name=user_dataset_folder_name,
                                                                         output_fn=output_fn)
        local_model = copy.deepcopy(model)
        local_model.train()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY,
                                    momentum=0.9)
        train_objects = TrainObjects(model=local_model, loader=train_loader, optimizer=optimizer)

        train_loss, train_acc, test_loss, test_acc = train_model(train_objects=train_objects,
                                                                 validation_loader=validation_loader,
                                                                 test_loader=test_loader,
                                                                 dp_params=dp_params,
                                                                 train_params=train_params,
                                                                 log2wandb=False, show_pbar=False)
        epoch_train_loss += train_loss / len(train_user_list)
        epoch_train_acc += train_acc / len(train_user_list)
        epoch_test_loss += test_loss / len(train_user_list)
        epoch_test_acc += test_acc / len(train_user_list)

        for n, p in local_model.named_parameters():
            params[n] += p.data

        # user_pbar.set_description(f'user train {u} loss {train_loss}'
        #                           f' acc {train_acc} test loss {test_loss} test acc {test_acc}')

    # average parameters
    for n, p in params.items():
        params[n] = p / Config.NUM_CLIENT_AGG

    # update new parameters
    # TODO batchnorm
    model.load_state_dict(params)

    return epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc


def federated_train_model(model, train_user_list, validation_user_list,
                          test_user_list,
                          num_epochs,
                          internal_train_params: TrainParams,
                          dp_params=None,
                          output_fn=lambda s: None):

    epoch_pbar = tqdm(range(num_epochs))
    for epoch in epoch_pbar:
        epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc = \
            federated_train_single_epoch(model=model,
                                         train_user_list=train_user_list,
                                         num_internal_epochs=num_internal_epochs,
                                         add_dp_noise_during_training=add_dp_noise_during_training,
                                         dp_params=dp_params)
        model.eval()
        val_loss, val_acc = 0, 0
        for u in validation_user_list:
            validation_loader = init_data_loaders(datasets_folder_name=os.path.join(Config.WINDOWED_DATA_DIR, u),
                                                  datasets=['validation'],
                                                  output_fn=lambda s: None)
            loss, acc = run_single_epoch(TrainObjects(loader=validation_loader, model=model))
            val_loss += loss / len(validation_user_list)
            val_acc += acc / len(validation_user_list)

        # wandb_log(epoch, train_loss=epoch_train_loss, train_acc=epoch_train_acc,
        #           test_loss=epoch_test_loss, test_acc=epoch_test_acc)

        epoch_pbar.set_description(f'federated global epoch {epoch} '
                                   f'train_loss {epoch_train_loss}, train_acc {epoch_train_acc} '
                                   f'test_loss {epoch_test_loss},  test_acc {epoch_test_acc} '
                                   f'val set loss {val_loss} val set acc {val_acc}')

    # Test Eval
    test_loss, test_acc = 0, 0
    for u in test_user_list:
        test_loader = init_data_loaders(datasets_folder_name=os.path.join(Config.WINDOWED_DATA_DIR, u),
                                        datasets=['test'],
                                        output_fn=lambda s: None)
        train_objects = TrainObjects(model=model, loader=test_loader)
        loss, acc = run_single_epoch(train_objects=train_objects)
        test_loss += loss / len(test_user_list)
        test_acc += acc / len(test_user_list)

    output_fn(f'Federated Train Finished Test Loss {test_loss} Test Acc {test_acc}')


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
                          output_fn=logger.info)


if __name__ == '__main__':
    main()
