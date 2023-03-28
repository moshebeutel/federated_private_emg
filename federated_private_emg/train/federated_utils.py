import copy
import os

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


def create_public_dataset(public_user: str or list[str]):
    user_dataset_folder_name = os.path.join(Config.WINDOWED_DATA_DIR, public_user) if isinstance(public_user, str) else\
        [os.path.join(Config.WINDOWED_DATA_DIR, pu) for pu in public_user]
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

    # print(net.__dict__)
    # print(net._forward_hooks)
    # print('Before Extend', net)
    net = extend(net)
    # print(net._forward_hooks)
    # print(net.__dict__)
    # for name,child in net.named_children():
    #     print(name, child.__dict__)
    # raise Exception
    # print('Before adding gep', net)
    # net.gep = gep
    # print('After adding gep', net)
    # raise Exception
    # print(net.gep.selected_bases_list)
    # print(net.gep.__dict__.keys())
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
    # net.gep.selected_bases_list = []
    gep.num_params = num_params
    # print(gep.num_param_list)
    # print(net.gep.num_param_list)
    #
    #
    # print(gep)
    # print(net.gep)

    # print(net.__dict__)
    # raise Exception
    loss_fn = extend(loss_fn)

    # gep.net_wrapper = NetWrapper(net)
    return net, loss_fn, gep


def federated_train_single_epoch(model, loss_fn, train_user_list, train_params: TrainParams, dp_params: DpParams = None,
                                 attach_gep_to_model_fn=None, output_fn=lambda s: None):
    epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc = 0.0, 0.0, 0.0, 0.0
    num_clients = len(train_user_list)
    optimizer = torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY,
                                momentum=0.9)
    pbar = tqdm(enumerate(train_user_list), desc='Iteration loop', leave=False)
    for i, u in pbar:
        # u = train_user_list[i]
        user_dataset_folder_name = os.path.join(Config.WINDOWED_DATA_DIR, u)
        test_loader, validation_loader, train_loader = init_data_loaders(datasets_folder_name=user_dataset_folder_name,
                                                                         output_fn=output_fn)
        local_model = copy.deepcopy(model)
        device = next(model.parameters()).device
        local_model = local_model.to(device)
        assert Config.USE_GEP == (attach_gep_to_model_fn is not None), f'USE_GEP = {Config.USE_GEP} but ' \
                                                                       f'attach_gep_to_model_fn {attach_gep_to_model_fn}'
        if Config.USE_GEP:
            local_model, loss_fn, gep = attach_gep_to_model_fn(local_model)
        local_model.train()

        # model.train()
        if i % Config.NUM_CLIENT_AGG == 0:
            optimizer.zero_grad()
            if Config.USE_GEP:
                gep.get_anchor_space(local_model, loss_func=loss_fn)
            for p in model.parameters():
                p.grad = torch.zeros_like(p)
        for p in local_model.parameters():
            p.grad = torch.zeros_like(p)
        train_objects = TrainObjects(model=local_model, loader=train_loader, criterion=loss_fn)
        user_loss, user_acc = 0.0, 0.0
        # internal train
        for _ in range(Config.NUM_INTERNAL_EPOCHS):
            loss, acc = \
                run_single_epoch_keep_grads(train_objects=train_objects, batch_size=train_params.batch_size,
                                            gep=gep if Config.USE_GEP else None)
            user_loss += loss / Config.NUM_INTERNAL_EPOCHS
            user_acc += acc / Config.NUM_INTERNAL_EPOCHS

        epoch_train_loss += user_loss / num_clients
        epoch_train_acc += user_acc / num_clients

        # dp
        if Config.USE_GEP:
            for p, p_local in zip(model.parameters(), local_model.parameters()):
                # Clip gradients and sum on global model
                p.grad += p_local.grad

            # batch_grad_list = []
            # for p in model.parameters():
            #     batch_grad_list.append(p.grad_batch.reshape(p.grad_batch.shape[0], -1))
            #     del p.grad_batch
            # clipped_theta, residual_grad, target_grad = gep(flatten_tensor(batch_grad_list))
            # theta_noise = torch.normal(0, Config.DP_SIGMA * Config.GEP_CLIP0 / Config.BATCH_SIZE,
            #                            size=clipped_theta.shape,
            #                            device=clipped_theta.device)
            # grad_noise = torch.normal(0, Config.DP_SIGMA * Config.GEP_CLIP1 / Config.BATCH_SIZE,
            #                           size=residual_grad.shape,
            #                           device=residual_grad.device)
            # clipped_theta += theta_noise
            # residual_grad += grad_noise
            #
            # noisy_grad = gep.get_approx_grad(clipped_theta) + residual_grad
            #
            # offset = 0
            # for p in model.parameters():
            #     shape = p.grad.shape
            #     numel = p.grad.numel()
            #     p.grad.data = noisy_grad[offset:offset + numel].view(shape)
            #     offset += numel

        else:
            grad_norm = calc_grad_norm(local_model)
            for p, p_local in zip(model.parameters(), local_model.parameters()):
                # Clip gradients and sum on global model
                p.grad += (p_local.grad / max(1, grad_norm / dp_params.dp_C))
                # Add DP noise to gradients
                noise = torch.normal(mean=0, std=dp_params.dp_sigma * dp_params.dp_C, size=p.size(), device=p.device)
                p.grad += noise

        del local_model

        if i % Config.NUM_CLIENT_AGG == (Config.NUM_CLIENT_AGG - 1):
            for p in model.parameters():
                # average grads
                p.grad /= Config.NUM_CLIENT_AGG
            # update parameters backwards
            optimizer.step()
        pbar.set_description(f"Iteration {i}. User {u}. Epoch running loss {epoch_train_loss}."
                             f" Epoch running acc {epoch_train_acc}")
        # user_pbar.set_description(f'user train {u} loss {train_loss}'
        #                           f' acc {train_acc} test loss {test_loss} test acc {test_acc}')

    # average parameters
    # for n, p in params.items():
    #     # params[n] = p / Config.NUM_CLIENT_AGG
    #     params[n] = p / num_clients

    # update new parameters
    # model.load_state_dict(params)
    return epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc


def federated_train_model(model, loss_fn, train_user_list, validation_user_list, test_user_list, num_epochs,
                          internal_train_params: TrainParams, dp_params=None, attach_gep_to_model_fn=None,
                          log2wandb=False,
                          output_fn=lambda s: None):
    eval_params = TrainParams(epochs=1, batch_size=internal_train_params.batch_size)
    epoch_pbar = tqdm(range(num_epochs), desc='Epoch Loop')
    for epoch in epoch_pbar:
        epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc = \
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
