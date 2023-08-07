import gc
import logging
import os
import random
import torch
import wandb
from common import utils
from common.config import Config
from common.utils import init_data_loaders, labels_to_consecutive, \
    create_toy_data, USERS_BIASES, USERS_VARIANCES, CIFAR10_CLASSES_NAMES, get_users_list_for_class
from differential_privacy.accountant_utils import accountant_params_string
from fed_priv_models.model_factory import init_model
from fed_priv_models.gep import GEP
from fed_priv_models.pFedGP.utils import build_tree, eval_model
from train.params import TrainParams
from train.train_utils import run_single_epoch, run_single_epoch_keep_grads, gep_batch, sgd_dp_batch, calc_metrics
import pandas as pd

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

        public_targets = public_targets if Config.CIFAR_DATA \
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
                                 gep=None,
                                 GPs=None,
                                 output_fn=lambda s: None):
    epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc = 0.0, 0.0, 0.0, 0.0
    sample_fn = random.choices if Config.SAMPLE_CLIENTS_WITH_REPLACEMENT else random.sample
    clients_in_epoch = sample_fn(train_user_list, k=Config.NUM_CLIENT_AGG)

    optimizer.zero_grad()

    if Config.USE_GEP:
        assert Config.USE_GEP == (gep is not None), f'USE_GEP = {Config.USE_GEP} but gep = {gep}'
        # gep.get_anchor_space(model, loss_func=loss_fn)
    if Config.PUBLIC_USERS_CONTRIBUTE_TO_LEARNING:
        clients_in_epoch = [*utils.public_users, *clients_in_epoch]

    num_clients_in_epoch = len(clients_in_epoch)

    for p in model.parameters():
        p.grad = torch.zeros_like(p, device=p.device)
        p.grad_batch = torch.zeros((num_clients_in_epoch, *p.grad.shape), device=p.device)

    # pbar = tqdm(enumerate(clients_in_epoch), desc='Iteration loop')
    # for i, u in pbar:
    for i, u in enumerate(clients_in_epoch):
        user_dataset_folder_name = os.path.join(Config.WINDOWED_DATA_DIR, u)

        train_loader = init_data_loaders(datasets_folder_name=user_dataset_folder_name, datasets=['train'],
                                         datasize=Config.PRIVATE_TRAIN_DATA_SIZE,
                                         output_fn=lambda s: None)  # output_fn)

        local_model = init_model()
        for lp, p in zip(local_model.parameters(), model.parameters()):
            lp.data = p.data

        local_optimizer = torch.optim.SGD(local_model.parameters(), lr=Config.LEARNING_RATE,
                                          weight_decay=Config.WEIGHT_DECAY, momentum=0.9)
        local_model.train()

        user_loss, user_acc = 0.0, 0.0

        # if Config.USE_GP:
        #     assert GPs is not None, f'Config.USE_GP={Config.USE_GP} but GPs is None'
        #     assert u in GPs, f'User {u} is not in epoch users {GPs.keys()}'
        #     # build tree at each step
        #     GPs[u], label_map, _, __ = build_tree(gp=GPs[u], net=local_model, loader=train_loader)
        #     GPs[u].train()

        loss, acc, _, local_model, local_optimizer = \
            run_single_epoch_keep_grads(model=local_model, optimizer=local_optimizer,
                                        loader=train_loader, criterion=loss_fn,
                                        batch_size=train_params.batch_size,
                                        gep=gep if Config.USE_GEP else None, gp=GPs[u] if Config.USE_GP else None)

        output_fn(f'client No.:{i} in epoch user:{u}-loss={loss},acc={acc}')

        # pull gradients from user
        for p, lp in zip(model.parameters(), local_model.parameters()):
            p.grad_batch[i] = lp.grad.data
            if i < Config.NUM_CLIENTS_PUBLIC - 1:
                p.grad.data += lp.grad.data.squeeze() / Config.NUM_CLIENTS_PUBLIC

        if Config.USE_GEP and i == len(gep.public_users) - 1:
            gep.get_anchor_space(model, loss_func=loss_fn)

        user_loss += loss / Config.NUM_INTERNAL_EPOCHS
        user_acc += acc / Config.NUM_INTERNAL_EPOCHS

        epoch_train_loss += user_loss / num_clients_in_epoch
        epoch_train_acc += user_acc / num_clients_in_epoch

        # Erase local train resources
        if Config.USE_GP:
            del GPs[u].tree
            GPs[u].tree = None
        del local_model, train_loader, local_optimizer, user_loss, user_acc, loss, acc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # pbar.set_description(f"Iteration {i}. User {u}. Epoch running loss {epoch_train_loss}."
        #                      f" Epoch running acc {epoch_train_acc}")

    if Config.USE_GEP:
        assert Config.USE_SGD_DP is False, 'Use GEP or SGD_DP. Not both'
        gep_batch(accumulated_grads=None, gep=gep, model=model, batchsize=Config.NUM_CLIENT_AGG)
    else:
        sgd_dp_batch(model=model, batchsize=Config.NUM_CLIENT_AGG)

    optimizer.step()
    return epoch_train_loss, epoch_train_acc, model


def federated_train_model(model, loss_fn, train_user_list, validation_user_list, test_user_list, num_epochs,
                          internal_train_params: TrainParams,
                          gep: GEP = None,
                          GPs=None,
                          log2wandb=False,
                          output_fn=lambda s: None):
    assert Config.USE_GEP == (gep is not None), f'USE_GEP = {Config.USE_GEP} but gep = {gep}'
    eval_params = TrainParams(epochs=1, batch_size=-1)
    # eval_params = TrainParams(epochs=1, batch_size=internal_train_params.batch_size)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=Config.GLOBAL_LEARNING_RATE * Config.NUM_CLIENT_AGG,
                                weight_decay=Config.WEIGHT_DECAY,
                                momentum=0.9)

    best_epoch_validation_acc = 0.0
    acc_decrease_count = 0
    best_model = init_model()

    # epoch_pbar = tqdm(range(num_epochs), desc='Epoch Loop')
    for epoch in range(num_epochs):
        model.train()
        output_fn(f'************      Epoch {epoch}         ******************')
        epoch_train_loss, epoch_train_acc, model = \
            federated_train_single_epoch(model=model, loss_fn=loss_fn, optimizer=optimizer,
                                         train_user_list=train_user_list,
                                         train_params=internal_train_params,
                                         gep=gep, GPs=GPs, output_fn=lambda s: None)  # output_fn)

        model.eval()
        val_losses, val_accs = [], {}
        if Config.USE_GP:
            val_results, labels_vs_preds_val = eval_model(model=model, GPs=GPs, split="validation",
                                                          users=utils.validation_user_list)
            val_losses = [val['loss'] for val in val_results.values()]
            val_accs_list = [100.0 * val['correct'] / val['total'] for val in val_results.values()]
            val_accs = {u: acc for (u, acc) in zip(val_results.keys(), val_accs_list)}

        else:

            for i, u in enumerate(validation_user_list):
                validation_loader = init_data_loaders(datasets_folder_name=os.path.join(Config.WINDOWED_DATA_DIR, u),
                                                      datasize=Config.BATCH_SIZE * 4,
                                                      datasets=['validation'],
                                                      output_fn=lambda s: None)
                loss, acc = run_single_epoch(loader=validation_loader, model=model, criterion=loss_fn,
                                             train_params=eval_params)

                val_losses.append(float(loss))
                val_accs[u] = float(acc)
                del loss, acc, validation_loader

        t_losses = torch.tensor(val_losses)
        t_accs = torch.tensor(list(val_accs.values()), dtype=torch.float)
        val_loss = t_losses.mean()
        val_acc = t_accs.mean()
        val_loss_std = t_losses.std()
        val_acc_std = t_accs.std()

        # epoch_pbar.set_description(f'federated global epoch {epoch} '
        #                            f'train_loss {epoch_train_loss}, train_acc {epoch_train_acc} '
        #                            f'val set loss {val_loss} val set acc {val_acc}')

        if not Config.USE_GP:
            loss_str = f'federated global epoch {epoch} train_loss {epoch_train_loss} val set loss {val_loss} std {val_loss_std}'
            acc_str = f' train_acc {epoch_train_acc} val set acc {val_acc} std {val_acc_std}'
        else:
            loss_str = f'federated global epoch {epoch} val set loss {val_loss} std {val_loss_std}'
            acc_str = f' val set acc {val_acc} std {val_acc_std}'

        output_fn(loss_str if Config.TOY_STORY else loss_str + acc_str)

        # logging.debug(acc_per_cls_string(user_accuracies_dict=val_accs, user_list=validation_user_list))

        # Release memory
        del t_accs, t_losses, val_losses, val_accs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # output_fn([f'{cls}, {CIFAR10_CLASSES_NAMES[cls]}' for cls in utils.CLASSES_OF_PUBLIC_USERS])

        if val_acc > best_epoch_validation_acc:
            best_epoch_validation_acc = val_acc
            acc_decrease_count = 0
            for bp, p in zip(best_model.parameters(), model.parameters()):
                bp.data = p.data
            logging.info(f'new best acc {best_epoch_validation_acc}')
        else:
            acc_decrease_count += 1
            logging.warning(f'acc_decrease_count {acc_decrease_count}')

        if log2wandb:
            log_dict = {'epoch_validation_loss': val_loss,
                        'epoch_validation_acc': val_acc,
                        'epoch_validation_loss_std': val_loss_std,
                        'epoch_validation_acc_std': val_acc_std,
                        'best_epoch_validation_acc': best_epoch_validation_acc
                        }

            if not Config.USE_GP:
                log_dict.update({'epoch_train_loss': epoch_train_loss,
                                 'epoch_train_acc': epoch_train_acc})

            if Config.USE_GEP:
                errs = torch.tensor(gep.approx_errors)
                log_dict.update({'epoch_gep_avg_approx_error': errs.mean(), 'epoch_gep_std_approx_error': errs.std()})

            wandb.log(log_dict)

        if acc_decrease_count > Config.EARLY_STOP_INCREASING_LOSS_COUNT:
            logging.warning(f'Accuracy decreases for {acc_decrease_count} rounds. Quit.')
            if Config.USE_GEP:
                output_fn(accountant_params_string())
            break

    # Test Eval
    if Config.TEST_AT_END:
        best_model.eval()
        test_loss, test_acc = 0, 0
        test_accuracies = {}

        if Config.USE_GP:
            test_results, labels_vs_preds_test = eval_model(model=best_model, GPs=GPs, split="test",
                                                            users=utils.test_user_list)
            test_loss = torch.mean(torch.tensor([test['loss'] for test in test_results.values()]))
            test_accs_list = [test['correct'] / test['total'] for test in test_results.values()]
            test_accuracies = {u: acc for (u, acc) in zip(test_results.keys(), test_accs_list)}
            test_acc = 100.0 * torch.mean(torch.tensor(test_accs_list))

        else:
            for u in test_user_list:
                test_loader = init_data_loaders(datasets_folder_name=os.path.join(Config.WINDOWED_DATA_DIR, u),
                                                datasize=Config.BATCH_SIZE * 4,
                                                datasets=['test'],
                                                output_fn=lambda s: None)
                loss, acc = run_single_epoch(model=best_model, loader=test_loader, criterion=loss_fn,
                                             train_params=eval_params)
                test_loss += loss / len(test_user_list)
                test_acc += acc / len(test_user_list)
                test_accuracies[u] = acc

        if Config.WRITE_TO_WANDB:
            wandb.log({'test_acc':test_acc})
            new_wandb_row = pd.DataFrame({**{k:v for (k,v) in wandb.config.items()
                                             if k not in ['app_config_dict', 'sweep_id']},
                                          **{'best_epoch_validation_acc': best_epoch_validation_acc,
                                             'test_acc': test_acc}}, index=[0])
            sweep_id = wandb.config["sweep_id"]
            sweep_csv_path = f'{Config.SWEEP_RESULTS_DIR}/{sweep_id}.csv'
            new_wandb_row.to_csv(sweep_csv_path, mode='a', index=False, header=True)
        # output_fn(acc_per_cls_string(user_accuracies_dict=test_accuracies, user_list=test_user_list))
        if Config.USE_GEP:
            output_fn(accountant_params_string())
        output_fn(f'Test Finished. Test Loss {test_loss} Test Acc {test_acc}')
    output_fn(f'Federated Train Finished')


def acc_per_cls_string(user_accuracies_dict: dict[str], user_list: list[str]) -> str:
    s = '\nAccuracies per class (mean,std):'
    for cls, cls_name in enumerate(CIFAR10_CLASSES_NAMES):
        users_for_class = get_users_list_for_class(cls, user_list)
        user_accuracies = torch.tensor([user_accuracies_dict[u] for u in users_for_class], dtype=torch.float)
        m = '%.3f' % (user_accuracies.mean().item())
        std = '%.3f' % (user_accuracies.std().item())
        s += f'\nNo.{cls},{cls_name}:({m},{std})'
    return s
