import copy
import gc
import os
import random

import torch
import wandb

from common.config import Config
from common.utils import init_data_loaders, labels_to_consecutive, \
    create_toy_data, USERS_BIASES, USERS_VARIANCES
from differential_privacy.params import DpParams
from fed_priv_models.gep import GEP
from train.params import TrainParams
from train.train_objects import TrainObjects
from train.train_utils import run_single_epoch, run_single_epoch_keep_grads, gep_batch, sgd_dp_batch


def create_public_dataset(public_users: str or list[str]):
    print('Create public dataset for users:', public_users)
    if not isinstance(public_users, list):
        public_users = [public_users]
    if Config.TOY_STORY:
        public_user_input_list, public_user_targets_list = [], []
        for u in public_users:
            bias = USERS_BIASES[u]
            variance = USERS_VARIANCES[u]
            public_user_input, public_user_target = \
                create_toy_data(data_size=Config.GEP_PUBLIC_DATA_SIZE, bias=bias, variance=variance)
            # print(public_user_input.shape, public_user_target.shape)
            public_user_input = public_user_input.unsqueeze(dim=0)
            public_user_target = public_user_target.unsqueeze(dim=0)
            # print(public_user_input.shape, public_user_target.shape)
            public_user_input_list.append(public_user_input)
            public_user_targets_list.append(public_user_target)
        public_inputs = torch.vstack(public_user_input_list)
        public_targets = torch.vstack(public_user_targets_list)
        # print(public_inputs.shape, public_targets.shape)
        public_inputs = torch.swapdims(public_inputs, 0, 1)
        public_targets = torch.swapdims(public_targets, 0, 1)
        # print(public_inputs.shape, public_targets.shape)

        public_inputs = torch.vstack([public_inputs[i] for i in range(public_inputs.shape[0])])
        public_targets = torch.vstack([public_targets[i] for i in range(public_targets.shape[0])])

        # print(public_inputs.shape, public_targets.shape)

    else:
        # user_dataset_folder_name = os.path.join(Config.WINDOWED_DATA_DIR, public_users) if isinstance(public_users,
        #                                                                                               str) else \
        #     [os.path.join(Config.WINDOWED_DATA_DIR, pu) for pu in public_users]
        # public_loader = init_data_loaders(datasets_folder_name=user_dataset_folder_name, datasets=['train'])
        public_inputs_list, public_targets_list = [], []
        for u in public_users:
            print('Creating public data for:', u)
            user_dataset_folder_name = os.path.join(Config.WINDOWED_DATA_DIR, u)
            public_loader = init_data_loaders(datasets_folder_name=user_dataset_folder_name,
                                              datasize=Config.GEP_PUBLIC_DATA_SIZE, datasets=['train'])
            for _ in range(int(Config.GEP_PUBLIC_DATA_SIZE / Config.BATCH_SIZE)):
                inputs, targets = next(iter(public_loader))
                inputs = inputs.unsqueeze(dim=0)
                public_inputs_list.append(inputs)
                public_targets_list.append(targets)
                del inputs, targets
            del public_loader
            gc.collect()

        public_inputs = torch.vstack(public_inputs_list)
        public_targets = torch.vstack(public_targets_list)

        del public_inputs_list, public_targets_list
        gc.collect()

        public_targets = public_targets if Config.CIFAR10_DATA \
            else labels_to_consecutive(public_targets).squeeze().long()
        # print('public data shape', public_inputs.shape, public_targets.shape)
        # print(public_inputs.shape, public_targets.shape)
        public_inputs = torch.swapdims(public_inputs, 0, 1)
        public_targets = torch.swapdims(public_targets, 0, 1)
        # print(public_inputs.shape, public_targets.shape)

        public_inputs = torch.vstack([public_inputs[i] for i in range(public_inputs.shape[0])])
        public_targets = torch.vstack(
            [public_targets[i].unsqueeze(dim=1) for i in range(public_targets.shape[0])]).squeeze()

        # print(public_inputs.shape, public_targets.shape)

    return public_inputs.float(), public_targets


def federated_train_single_epoch(model, loss_fn, optimizer, train_user_list, train_params: TrainParams,
                                 dp_params: DpParams = None,
                                 gep=None,
                                 attach_gep_to_model_fn=None,
                                 output_fn=lambda s: None):
    epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc = 0.0, 0.0, 0.0, 0.0
    # num_clients_in_epoch = len(train_user_list)
    num_clients_in_epoch = Config.NUM_CLIENT_AGG
    # clients_in_epoch = random.choices(train_user_list, k=num_clients_in_epoch)
    sample_fn = random.choices if Config.SAMPLE_CLIENTS_WITH_REPLACEMENT else random.sample
    clients_in_epoch = sample_fn(train_user_list, k=num_clients_in_epoch)
    device = next(model.parameters()).device

    optimizer.zero_grad()

    if Config.USE_GEP:
        assert Config.USE_GEP == (gep is not None), f'USE_GEP = {Config.USE_GEP} but gep = {gep}'
        gep.get_anchor_space(model, loss_func=loss_fn)

    for p in model.parameters():
        p.grad = torch.zeros_like(p, device=p.device)
        p.grad_batch = torch.zeros((num_clients_in_epoch, *p.grad.shape), device=p.device)

    # pbar = tqdm(enumerate(clients_in_epoch), desc='Iteration loop')
    # for i, u in pbar:
    for i, u in enumerate(clients_in_epoch):
        user_dataset_folder_name = os.path.join(Config.WINDOWED_DATA_DIR, u)

        train_loader = init_data_loaders(datasets_folder_name=user_dataset_folder_name, datasets=['train'],
                                         datasize=Config.PRIVATE_TRAIN_DATA_SIZE,
                                         output_fn=output_fn)
        local_model = copy.deepcopy(model).to(device)

        local_optimizer = torch.optim.SGD(local_model.parameters(), lr=Config.LEARNING_RATE,
                                          weight_decay=Config.WEIGHT_DECAY, momentum=0.9)
        local_model.train()

        user_loss, user_acc = 0.0, 0.0

        # Internal train

        # p_inner_bar = tqdm(range(Config.NUM_INTERNAL_EPOCHS), desc='Internal loop', leave=False)
        # for _ in p_inner_bar:
        for internal_epoch in range(Config.NUM_INTERNAL_EPOCHS):
            loss, acc, _, local_model, local_optimizer = \
                run_single_epoch_keep_grads(model=local_model, optimizer=local_optimizer,
                                            loader=train_loader, criterion=loss_fn,
                                            batch_size=train_params.batch_size,
                                            gep=gep if Config.USE_GEP else None)

            if Config.TOY_STORY:
                print('User', i, u, 'Bias: ', '%.3f' % USERS_BIASES[u], 'Variance: ', '%.3f' % USERS_VARIANCES[u],
                      'internal epoch', internal_epoch, 'Loss', '%.3f' % loss)
            else:
                print('User', i, u, 'internal epoch', internal_epoch, 'Loss', '%.3f' % loss, 'acc', acc)
            # if Config.WRITE_TO_WANDB:
            #     wandb.log({
            #         f'User {u} Train Loss': loss
            #     })

            for p, lp in zip(model.parameters(), local_model.parameters()):
                # p.grad.data += (lp.grad.data / num_clients_in_epoch)
                p.grad_batch[i] = lp.grad.data
                del lp.grad
                # print('p.grad', p.grad)
                # print('p.grad_batch', p.grad_batch)

            user_loss += loss / Config.NUM_INTERNAL_EPOCHS
            user_acc += acc / Config.NUM_INTERNAL_EPOCHS

            del loss, acc

        epoch_train_loss += user_loss / num_clients_in_epoch
        epoch_train_acc += user_acc / num_clients_in_epoch

        del local_model, train_loader, local_optimizer, user_loss, user_acc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # pbar.set_description(f"Iteration {i}. User {u}. Epoch running loss {epoch_train_loss}."
        #                      f" Epoch running acc {epoch_train_acc}")

    if Config.USE_GEP:
        assert Config.USE_SGD_DP is False, 'Use GEP or SGD_DP. Not both'
        gep_batch(accumulated_grads=None, gep=gep, model=model, batchsize=num_clients_in_epoch)
    elif Config.ADD_DP_NOISE:
        sgd_dp_batch(model=model, batchsize=num_clients_in_epoch)

    optimizer.step()
    return epoch_train_loss, epoch_train_acc, model


def federated_train_model(model, loss_fn, train_user_list, validation_user_list, test_user_list, num_epochs,
                          internal_train_params: TrainParams, dp_params=None,
                          gep: GEP = None,
                          # attach_gep_to_model_fn=None,
                          log2wandb=False,
                          output_fn=lambda s: None):
    assert Config.USE_GEP == (gep is not None), f'USE_GEP = {Config.USE_GEP} but gep = {gep}'
    eval_params = TrainParams(epochs=1, batch_size=-1)
    # eval_params = TrainParams(epochs=1, batch_size=internal_train_params.batch_size)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=Config.GLOBAL_LEARNING_RATE * Config.NUM_CLIENT_AGG,
                                weight_decay=Config.WEIGHT_DECAY,
                                momentum=0.9)

    best_loss = 100000
    loss_increase_count = 0

    # assert Config.USE_GEP == (attach_gep_to_model_fn is not None), \
    #     f'USE_GEP = {Config.USE_GEP} but ' \
    #     f'attach_gep_to_model_fn {attach_gep_to_model_fn}'
    # gep = None
    # if Config.USE_GEP:
    #     model, loss_fn, gep = attach_gep_to_model_fn(model)

    # epoch_pbar = tqdm(range(num_epochs), desc='Epoch Loop')
    for epoch in range(num_epochs):
        model.train()

        epoch_train_loss, epoch_train_acc, model = \
            federated_train_single_epoch(model=model, loss_fn=loss_fn, optimizer=optimizer,
                                         train_user_list=train_user_list,
                                         train_params=internal_train_params,
                                         dp_params=dp_params,
                                         gep=gep)

        model.eval()
        val_losses, val_accs = [], []
        for i, u in enumerate(validation_user_list):
            validation_loader = init_data_loaders(datasets_folder_name=os.path.join(Config.WINDOWED_DATA_DIR, u),
                                                  datasize=Config.BATCH_SIZE * 4,
                                                  datasets=['validation'],
                                                  output_fn=lambda s: None)
            loss, acc = run_single_epoch(loader=validation_loader, model=model, criterion=loss_fn,
                                         train_params=eval_params)

            if Config.TOY_STORY:
                print('User', i, u, 'Bias: ', '%.3f' % USERS_BIASES[u], 'Variance: ', '%.3f' % USERS_VARIANCES[u],
                      'Validation Loss', '%.3f' % loss)
            else:
                print('User', i, u, 'Validation Loss', '%.3f' % loss, 'acc', acc)
            val_losses.append(float(loss))
            val_accs.append(float(acc))
            # if log2wandb:
            #     wandb.log({
            #         f'User {u} Validation Loss': loss
            #     })

            # val_loss += float(loss) / float(len(validation_user_list))
            # val_acc += float(acc) / float(len(validation_user_list))

            t_losses = torch.tensor(val_losses)
            t_accs = torch.tensor(val_accs)
            val_loss = t_losses.mean()
            val_acc = t_accs.mean()
            val_loss_std = t_losses.std()
            val_acc_std = t_losses.std()
            # Release memory
            del loss, acc, validation_loader, t_accs, t_losses
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # epoch_pbar.set_description(f'federated global epoch {epoch} '
        #                            f'train_loss {epoch_train_loss}, train_acc {epoch_train_acc} '
        #                            f'val set loss {val_loss} val set acc {val_acc}')

        loss_str = f'federated global epoch {epoch} train_loss {epoch_train_loss} val set loss {val_loss} std {val_loss_std}'
        acc_str = f' train_acc {epoch_train_acc} val set acc {val_acc} std {val_acc_std}'

        if Config.TOY_STORY:
            print(loss_str)
        else:
            print(loss_str + acc_str)

        if log2wandb:
            wandb.log({
                'epoch_train_loss': epoch_train_loss,
                'epoch_train_acc': epoch_train_acc,
                'epoch_validation_loss': val_loss,
                'epoch_validation_acc': val_acc,
                'epoch_validation_loss_std': val_loss_std,
                'epoch_validation_acc_std': val_acc_std,
            })

        if val_loss < best_loss:
            best_loss = val_loss
            loss_increase_count = 0
        else:
            loss_increase_count += 1
            print('loss_increase_count', loss_increase_count)

        if loss_increase_count > Config.EARLY_STOP_INCREASING_LOSS_COUNT:
            break

    # Test Eval
    test_loss, test_acc = 0, 0
    for u in test_user_list:
        test_loader = init_data_loaders(datasets_folder_name=os.path.join(Config.WINDOWED_DATA_DIR, u),
                                        datasize=Config.BATCH_SIZE * 4,
                                        datasets=['test'],
                                        output_fn=lambda s: None)
        loss, acc = run_single_epoch(model=model, loader=test_loader, criterion=loss_fn, train_params=eval_params)
        test_loss += loss / len(test_user_list)
        test_acc += acc / len(test_user_list)

    output_fn(f'Federated Train Finished Test Loss {test_loss} Test Acc {test_acc}')
