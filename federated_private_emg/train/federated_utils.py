import copy
import os
from collections import OrderedDict

import torch
import wandb
from tqdm import tqdm

from common.config import Config
from common.utils import init_data_loaders, calc_grad_norm, flatten_tensor, labels_to_consecutive
from differential_privacy.params import DpParams
from fed_priv_models.pad_operators import NetWrapper
from train.params import TrainParams
from train.train_objects import TrainObjects
from train.train_utils import run_single_epoch, run_single_epoch_keep_grads
from fed_priv_models.gep import GEP, extend
import torch
from collections.abc import Callable


def create_public_dataset(public_users: str or list[str]):
    user_dataset_folder_name = os.path.join(Config.WINDOWED_DATA_DIR, public_users) if isinstance(public_users,
                                                                                                  str) else \
        [os.path.join(Config.WINDOWED_DATA_DIR, pu) for pu in public_users]
    public_loader = init_data_loaders(datasets_folder_name=user_dataset_folder_name, datasets=['validation'])
    # public_data = list(public_loader)
    #
    # public_inputs = torch.vstack([i[0] for i in public_data])
    # public_targets = torch.vstack([i[1] for i in public_data])
    public_inputs, public_targets = next(iter(public_loader))
    public_targets = labels_to_consecutive(public_targets).squeeze()
    print('public data shape', public_inputs.shape, public_targets.shape)
    return public_inputs.float(), public_targets.long()


def attach_gep(net: torch.nn.Module, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], num_bases: int,
               batch_size: int, clip0: float, clip1: float, power_iter: int,
               num_groups: int, public_inputs: torch.Tensor, public_targets: torch.Tensor):
    # print('\n==> Creating GEP class instance')
    gep = GEP(num_bases, batch_size, clip0, clip1, power_iter).cuda()
    ## attach auxiliary data to GEP instance

    public_inputs, public_targets = public_inputs.cuda(), public_targets.cuda()
    gep.public_inputs = public_inputs
    gep.public_targets = public_targets

    net = extend(net)

    num_params = 0
    np_list = []
    for p in net.parameters():
        num_params += p.numel()
        np_list.append(p.numel())

    def group_params(num_p, groups):
        assert groups >= 1

        p_per_group = num_p // groups
        num_param_list = [p_per_group] * (groups - 1)
        num_param_list = num_param_list + [num_p - sum(num_param_list)]
        return num_param_list

    # print(f'\n==> Dividing {num_params} parameters in to {num_groups} groups')
    gep.num_param_list = group_params(num_params, num_groups)

    gep.num_params = num_params

    loss_fn = extend(loss_fn)

    return net, loss_fn, gep


def federated_train_single_epoch(model, loss_fn, train_user_list, train_params: TrainParams, dp_params: DpParams = None,
                                 attach_gep_to_model_fn=None, output_fn=lambda s: None):
    epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc = 0.0, 0.0, 0.0, 0.0
    num_clients = len(train_user_list)

    device = next(model.parameters()).device
    model.train()

    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer = torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY,
                                momentum=0.9)
    optimizer.zero_grad()

    # pbar = tqdm(enumerate(train_user_list), desc='Iteration loop')
    # for i, u in pbar:
    for i, u in enumerate(train_user_list):
        user_dataset_folder_name = os.path.join(Config.WINDOWED_DATA_DIR, u)
        test_loader, validation_loader, train_loader = init_data_loaders(datasets_folder_name=user_dataset_folder_name,
                                                                         output_fn=output_fn)
        local_model = copy.deepcopy(model).to(device)

        assert Config.USE_GEP == (attach_gep_to_model_fn is not None), \
            f'USE_GEP = {Config.USE_GEP} but ' \
            f'attach_gep_to_model_fn {attach_gep_to_model_fn}'
        if Config.USE_GEP:
            local_model, loss_fn, gep = attach_gep_to_model_fn(local_model)
        local_optimizer = torch.optim.SGD(local_model.parameters(), lr=Config.LEARNING_RATE,
                                          weight_decay=Config.WEIGHT_DECAY, momentum=0.9)
        local_model.train()
        # local_optimizer.zero_grad()
        for p in local_model.parameters():
            p.grad = torch.zeros_like(p)
        if i % Config.NUM_CLIENT_AGG == 0:
            if Config.USE_GEP:
                local_optimizer.zero_grad()
                gep.get_anchor_space(local_model, loss_func=loss_fn)

        user_loss, user_acc = 0.0, 0.0

        # Internal train
        # init grads accumulator
        internal_train_grads = OrderedDict()
        for n, p in model.named_parameters():
            internal_train_grads[n] = torch.zeros_like(p.data)
        # p_inner_bar = tqdm(range(Config.NUM_INTERNAL_EPOCHS), desc='Internal loop', leave=False)
        # for _ in p_inner_bar:
        for internal_epoch in range(Config.NUM_INTERNAL_EPOCHS):
            loss, acc, epoch_grads, local_model, local_optimizer = \
                run_single_epoch_keep_grads(model=local_model, optimizer=local_optimizer,
                                            loader=train_loader, criterion=loss_fn,
                                            batch_size=train_params.batch_size,
                                            gep=gep if Config.USE_GEP else None)
            # print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
            # print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
            # print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
            # print('User', i, u, 'internal epoch', internal_epoch, 'Loss', loss, 'acc', acc)

            for n, p in model.named_parameters():
                internal_train_grads[n] += epoch_grads[n]

            user_loss += loss / Config.NUM_INTERNAL_EPOCHS
            user_acc += acc / Config.NUM_INTERNAL_EPOCHS

        epoch_train_loss += user_loss / num_clients
        epoch_train_acc += user_acc / num_clients

        for n, p in model.named_parameters():
            p.grad += (internal_train_grads[n] / num_clients)

        # del internal_train_grads
        # del local_model

        if i % Config.NUM_CLIENT_AGG == (Config.NUM_CLIENT_AGG - 1):
            for p in model.parameters():
                # average grads
                p.grad /= Config.NUM_CLIENT_AGG

        # pbar.set_description(f"Iteration {i}. User {u}. Epoch running loss {epoch_train_loss}."
        #                      f" Epoch running acc {epoch_train_acc}")
    optimizer.step()
    return epoch_train_loss, epoch_train_acc, model


def federated_train_model(model, loss_fn, train_user_list, validation_user_list, test_user_list, num_epochs,
                          internal_train_params: TrainParams, dp_params=None, attach_gep_to_model_fn=None,
                          log2wandb=False,
                          output_fn=lambda s: None):
    eval_params = TrainParams(epochs=1, batch_size=internal_train_params.batch_size)
    # epoch_pbar = tqdm(range(num_epochs), desc='Epoch Loop')
    for epoch in range(num_epochs):
        epoch_train_loss, epoch_train_acc, model = \
            federated_train_single_epoch(model=model, loss_fn=loss_fn,
                                         train_user_list=train_user_list,
                                         train_params=internal_train_params,
                                         dp_params=dp_params, attach_gep_to_model_fn=attach_gep_to_model_fn)

        model.eval()
        val_loss, val_acc = 0, 0
        for u in validation_user_list:
            validation_loader = init_data_loaders(datasets_folder_name=os.path.join(Config.WINDOWED_DATA_DIR, u),
                                                  datasets=['validation'],
                                                  output_fn=lambda s: None)
            loss, acc = run_single_epoch(TrainObjects(loader=validation_loader, model=model), train_params=eval_params)
            val_loss += loss / len(validation_user_list)
            val_acc += acc / len(validation_user_list)

        # epoch_pbar.set_description(f'federated global epoch {epoch} '
        #                            f'train_loss {epoch_train_loss}, train_acc {epoch_train_acc} '
        #                            f'val set loss {val_loss} val set acc {val_acc}')
        print(f'federated global epoch {epoch} '
              f'train_loss {epoch_train_loss}, train_acc {epoch_train_acc} '
              f'val set loss {val_loss} val set acc {val_acc}')

        if log2wandb:
            wandb.log({
                'epoch_train_loss': epoch_train_loss,
                'epoch_train_acc': epoch_train_acc,
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
