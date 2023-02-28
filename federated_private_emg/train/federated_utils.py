import copy
import os
from collections import OrderedDict

import torch
from tqdm import tqdm

import wandb
from common.config import Config
from common.utils import init_data_loaders, calc_grad_norm
from train.params import TrainParams
from train.train_objects import TrainObjects
from train.train_utils import train_model, run_single_epoch, run_single_epoch_return_grads


def federated_train_single_epoch(model, train_user_list,
                                 train_params: TrainParams,
                                 dp_params=None,
                                 output_fn=lambda s: None):
    # params = OrderedDict()
    # for n, p in model.named_parameters():
    #     params[n] = torch.zeros_like(p.data)
    epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc = 0.0, 0.0, 0.0, 0.0
    num_clients = len(train_user_list)
    optimizer = torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY,
                                momentum=0.9)
    for i, u in enumerate(train_user_list):
        # u = train_user_list[i]
        user_dataset_folder_name = os.path.join(Config.WINDOWED_DATA_DIR, u)
        test_loader, validation_loader, train_loader = init_data_loaders(datasets_folder_name=user_dataset_folder_name,
                                                                         output_fn=output_fn)
        local_model = copy.deepcopy(model)
        local_model.train()
        # optimizer = torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY,
        #                             momentum=0.9)
        model.train()
        if i % Config.NUM_CLIENT_AGG == 0:
            optimizer.zero_grad()
            for p in model.parameters():
                p.grad = torch.zeros_like(p)
        train_objects = TrainObjects(model=local_model, loader=train_loader)

        # train_loss, train_acc, test_loss, test_acc = train_model(train_objects=train_objects,
        #                                                          validation_loader=validation_loader,
        #                                                          test_loader=test_loader,
        #                                                          dp_params=dp_params,
        #                                                          train_params=train_params,
        #                                                          log2wandb=False, show_pbar=False)

        train_loss, train_acc = \
            run_single_epoch_return_grads(train_objects=train_objects, batch_size=train_params.batch_size)

        epoch_train_loss += train_loss / num_clients
        epoch_train_acc += train_acc / num_clients

        grad_norm = calc_grad_norm(local_model)
        for p, p_local in zip(model.parameters(), local_model.parameters()):
            # Clip gradients and sum on global model
            p.grad += (p_local.grad / max(1, grad_norm / dp_params.dp_C))

        del local_model

        # for n, p in local_model.named_parameters():
        #     params[n] += p.grad

        if i % Config.NUM_CLIENT_AGG == (Config.NUM_CLIENT_AGG - 1):

            for p in model.parameters():
                # Add DP noise to gradients
                noise = torch.normal(mean=0, std=dp_params.dp_sigma * dp_params.dp_C, size=p.size(), device=p.device)
                p.grad += noise

                # average grads
                p.grad /= Config.NUM_CLIENT_AGG

            # update parameters backwards
            optimizer.step()


        # user_pbar.set_description(f'user train {u} loss {train_loss}'
        #                           f' acc {train_acc} test loss {test_loss} test acc {test_acc}')

    # average parameters
    # for n, p in params.items():
    #     # params[n] = p / Config.NUM_CLIENT_AGG
    #     params[n] = p / num_clients

    # update new parameters
    # model.load_state_dict(params)

    return epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc


def federated_train_model(model, train_user_list, validation_user_list,
                          test_user_list,
                          num_epochs,
                          internal_train_params: TrainParams,
                          dp_params=None, log2wandb=False,
                          output_fn=lambda s: None):
    eval_params = TrainParams(epochs=1, batch_size=internal_train_params.batch_size)
    epoch_pbar = tqdm(range(num_epochs))
    for epoch in epoch_pbar:
        epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc = \
            federated_train_single_epoch(model=model,
                                         train_user_list=train_user_list,
                                         train_params=internal_train_params,
                                         dp_params=dp_params)
        model.eval()
        val_loss, val_acc = 0, 0
        for u in validation_user_list:
            validation_loader = init_data_loaders(datasets_folder_name=os.path.join(Config.WINDOWED_DATA_DIR, u),
                                                  datasets=['validation'],
                                                  output_fn=lambda s: None)
            loss, acc = run_single_epoch(TrainObjects(loader=validation_loader, model=model), train_params=eval_params)
            val_loss += loss / len(validation_user_list)
            val_acc += acc / len(validation_user_list)

        epoch_pbar.set_description(f'federated global epoch {epoch} '
                                   f'train_loss {epoch_train_loss}, train_acc {epoch_train_acc} '
                                   f'test_loss {epoch_test_loss},  test_acc {epoch_test_acc} '
                                   f'val set loss {val_loss} val set acc {val_acc}')

        if log2wandb:
            wandb.log({
                'epoch_train_loss': epoch_train_loss,
                'epoch_train_acc': epoch_train_acc,
                'epoch_test_loss': epoch_test_loss,
                'epoch_test_acc': epoch_test_acc,
                'epoch_validation_loss': val_loss,
                'epoch_validation_acc': val_acc,
            })

    # Test Eval
    test_loss, test_acc = 0, 0
    for u in test_user_list:
        test_loader = init_data_loaders(datasets_folder_name=os.path.join(Config.WINDOWED_DATA_DIR, u),
                                        datasets=['test'],
                                        output_fn=lambda s: None)
        train_objects = TrainObjects(model=model, loader=test_loader)
        loss, acc = run_single_epoch(train_objects=train_objects, train_params=eval_params)
        test_loss += loss / len(test_user_list)
        test_acc += acc / len(test_user_list)

    output_fn(f'Federated Train Finished Test Loss {test_loss} Test Acc {test_acc}')
