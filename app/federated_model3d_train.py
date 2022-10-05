import copy
import logging
import os
from collections import OrderedDict

import torch
from tqdm import tqdm

import wandb
from common import utils
from federated_private_emg.fed_priv_models.model3d import Model3d
from common.utils import init_data_loaders, SimpleGlobalCounter, wandb_log, run_single_epoch
from common.config import Config, WRITE_TO_WANDB


def user_internal_train(criterion, local_model, optimizer, train_loader, test_loader, num_epochs: int = 1):
    epoch_pbar = tqdm(range(num_epochs))
    for epoch in epoch_pbar:
        train_loss, train_acc = run_single_epoch(loader=train_loader,
                                                 model=local_model,
                                                 criterion=criterion,
                                                 optimizer=optimizer,
                                                 global_sample_counter=SimpleGlobalCounter())
        local_model.eval()
        test_loss, test_acc = run_single_epoch(loader=test_loader, model=local_model, criterion=criterion)
        local_model.train()
        epoch_pbar.set_description(f'user internal  epoch {epoch} loss {train_loss}'
                                   f' acc {train_acc} test loss {test_loss} test acc {test_acc}')

    return train_loss, train_acc, test_loss, test_acc


def federated_train_single_epoch(model, train_user_list, num_internal_epochs):
    params = OrderedDict()
    for n, p in model.named_parameters():
        params[n] = torch.zeros_like(p.data)
    epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc = 0.0, 0.0, 0.0, 0.0
    user_pbar = tqdm(range(len(train_user_list)))
    for i in user_pbar:
        u = train_user_list[i]
        user_dataset_folder_name = os.path.join(Config.WINDOWED_DATA_DIR, u)
        test_loader, train_loader = init_data_loaders(datasets_folder_name=user_dataset_folder_name,
                                                      output_fn=lambda s: None)
        local_model = copy.deepcopy(model)
        local_model.train()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY,
                                    momentum=0.9)
        train_loss, train_acc, test_loss, test_acc = user_internal_train(criterion=torch.nn.CrossEntropyLoss(),
                                                                         local_model=local_model, optimizer=optimizer,
                                                                         train_loader=train_loader,
                                                                         test_loader=test_loader,
                                                                         num_epochs=num_internal_epochs)
        epoch_train_loss += train_loss / len(train_user_list)
        epoch_train_acc += train_acc / len(train_user_list)
        epoch_test_loss += test_loss / len(train_user_list)
        epoch_test_acc += test_acc / len(train_user_list)

        for n, p in local_model.named_parameters():
            params[n] += p.data

        user_pbar.set_description(f'user train {u} loss {train_loss}'
                                  f' acc {train_acc} test loss {test_loss} test acc {test_acc}')

    # average parameters
    for n, p in params.items():
        params[n] = p / Config.NUM_CLIENT_AGG

    # update new parameters
    #TODO batchnorm
    model.load_state_dict(params)

    return epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc


def federated_train_model(model, train_user_list, validation_user_list, test_user_list, num_internal_epochs,
                          num_epochs, output_fn):
    epoch_pbar = tqdm(range(num_epochs))
    for epoch in epoch_pbar:
        epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc = \
            federated_train_single_epoch(model, train_user_list, num_internal_epochs)
        model.eval()
        val_loss, val_acc = 0, 0
        for u in validation_user_list:
            validation_loader, _ = init_data_loaders(datasets_folder_name=os.path.join(Config.WINDOWED_DATA_DIR, u),
                                                     x_test_filename='X_validation_windowed.pt',
                                                     y_test_filename='y_validation_windowed.pt',
                                                     output_fn=output_fn)
            loss, acc = run_single_epoch(loader=validation_loader, model=model, criterion=torch.nn.CrossEntropyLoss())
            val_loss += loss / len(validation_user_list)
            val_acc += acc / len(validation_user_list)

        wandb_log(epoch, train_loss=epoch_train_loss, train_acc=epoch_train_acc, test_acc=val_acc)

        epoch_pbar.set_description(f'federated global epoch {epoch} val {val_loss} val acc {val_acc}')

    # Test Eval
    test_loss, test_acc = 0, 0
    for u in test_user_list:
        test_loader, _ = init_data_loaders(datasets_folder_name=os.path.join(Config.WINDOWED_DATA_DIR, u),
                                           output_fn=print)
        loss, acc = run_single_epoch(loader=test_loader, model=model, criterion=torch.nn.CrossEntropyLoss())
        test_loss += loss / len(test_user_list)
        test_acc += acc / len(test_user_list)

    output_fn(f'Federated Train Finished Test Loss {test_loss} Test Acc {test_acc}')


def main():
    exp_name = utils.get_exp_name(os.path.basename(__file__)[:-3])
    logger = utils.config_logger(f'{exp_name}_logger',
                                 level=logging.INFO, log_folder='../log/')
    logger.info(exp_name)
    if WRITE_TO_WANDB:
        wandb.init(project="emg_gp_moshe", entity="emg_diff_priv", name=exp_name)
        # wandb.config.update({})

    model = Model3d(number_of_classes=Config.NUM_CLASSES, window_size=Config.WINDOW_SIZE, output_info_fn=logger.info,
                    output_debug_fn=logger.debug)

    federated_train_model(model=model, train_user_list=['03', '04', '05', '08', '09'], validation_user_list=['06'],
                          test_user_list=['07'], num_internal_epochs=Config.NUM_INTERNAL_EPOCHS,
                          num_epochs=Config.NUM_EPOCHS, output_fn=logger.info)


if __name__ == '__main__':
    main()

