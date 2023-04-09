from __future__ import annotations
import math
import copy
from collections import OrderedDict
from dataclasses import astuple

import torch
import wandb
from backpack import backpack
from backpack.extensions import BatchGrad
from tqdm import tqdm

# from common.config import Config
from common.utils import labels_to_consecutive, flatten_tensor
from differential_privacy.params import DpParams
from differential_privacy.utils import add_dp_noise, per_sample_gradient_fwd_bwd
from fed_priv_models.gep import GEP
from train.params import TrainParams
from train.train_objects import TrainObjects
from collections.abc import Callable
from common.config import Config
import matplotlib.pyplot as plt


def run_single_epoch_keep_grads(model, optimizer, loader, criterion,
                                batch_size: int, gep: GEP = None, use_dp_noise: bool = False) -> (float, float):
    assert model.training, 'Model not in train mode'
    running_loss, correct_counter, sample_counter, counter = 0., 0, 0, 0
    device = next(model.parameters()).device

    bench_model = copy.deepcopy(model)
    bench_model.train()

    # init grads accumulator
    accumulated_grads = OrderedDict()
    for ((n, p), (bn, bp)) in zip(model.named_parameters(), bench_model.named_parameters()):
        accumulated_grads[n] = torch.zeros_like(p.data)
        bp.data = copy.deepcopy(p.data)
        p.grad = torch.zeros_like(p.data)
        bp.grad = torch.zeros_like(bp.data)

    bench_optimizer = torch.optim.SGD(bench_model.parameters(), lr=Config.LEARNING_RATE,
                                      weight_decay=Config.WEIGHT_DECAY, momentum=0.9)

    if Config.TOY_STORY and Config.PLOT_GRADS:
        grads_plot_list = {}
        bench_grads_plot_list = {}

        params_plot_list = {}
        bench_params_plot_list = {}

        losses = []
        bench_losses = []
        print('Init')
        for ((n, p), (bn, bp)) in zip(model.named_parameters(), bench_model.named_parameters()):
            assert not (n in params_plot_list.keys())
            params_plot_list[n] = [p.data.detach()]

            assert not (bn in bench_params_plot_list.keys())
            bench_params_plot_list[bn] = [bp.data.detach()]

            assert not (n in grads_plot_list.keys())
            grads_plot_list[n] = [p.grad.data.detach()]

            assert not (bn in bench_grads_plot_list.keys())
            bench_grads_plot_list[bn] = [bp.grad.data.detach()]

            # print('@@@ params $$$')
            # print(n, p.data.detach(), bp.data.detach(), p.data.detach() - bp.data.detach())
            #
            # print('### grads &&&')
            # print(n, p.grad.data.detach(), bp.grad.data.detach(), p.grad.data.detach() - bp.grad.data.detach())

    for k, batch in enumerate(loader):

        curr_batch_size = batch[1].size(0)
        if curr_batch_size < batch_size:
            continue
        counter += 1

        sample_counter += curr_batch_size
        batch = (t.to(device) for t in batch)
        emg, labels = batch
        if Config.TOY_STORY:
            pass  # labels = torch.hstack([labels, torch.zeros_like(labels)])
        else:
            labels = labels_to_consecutive(labels).squeeze()
        optimizer.zero_grad()
        bench_optimizer.zero_grad()

        outputs = model(emg.float())
        bench_outputs = bench_model(emg.float())

        loss = criterion(outputs, labels)
        bench_loss = criterion(bench_outputs, labels)
        running_loss += float(loss)
        _, predicted = torch.max(outputs.data, 1)
        if model.training:
            if Config.USE_GEP:
                with backpack(BatchGrad()):
                    loss.backward()
            else:
                loss.backward()
            bench_loss.backward()

        if Config.USE_GEP:
            assert gep is not None, f'Config.USE_GEP={Config.USE_GEP} but gep object not provided. gep={gep}'
            batch_grad_list = []
            for p in model.parameters():
                if hasattr(p, 'grad_batch'):
                    batch_grad_list.append(p.grad_batch.reshape(p.grad_batch.shape[0], -1))
                    del p.grad_batch
            clipped_theta, residual_grad, target_grad = gep(flatten_tensor(batch_grad_list))

            clean_grad = gep.get_approx_grad(clipped_theta) + residual_grad

            if Config.ADD_DP_NOISE:
                # Perturbation
                theta_noise = torch.normal(0, Config.DP_SIGMA * Config.GEP_CLIP0 / Config.BATCH_SIZE,
                                           size=clipped_theta.shape,
                                           device=clipped_theta.device)
                grad_noise = torch.normal(0, Config.DP_SIGMA * Config.GEP_CLIP1 / Config.BATCH_SIZE,
                                          size=residual_grad.shape,
                                          device=residual_grad.device)

                # print('theta_noise', torch.linalg.norm(theta_noise))
                # print('grad_noise', torch.linalg.norm(grad_noise))
                clipped_theta += theta_noise
                residual_grad += grad_noise

            # print('clipped_theta', clipped_theta, torch.linalg.norm(clipped_theta))
            # print('residual_grad', residual_grad, torch.linalg.norm(residual_grad))
            # print('target_grad', target_grad, torch.linalg.norm(target_grad))
            # print('clean_grad', clean_grad, torch.linalg.norm(clean_grad))
            # print('gep.get_approx_grad(clipped_theta)', gep.get_approx_grad(clipped_theta))
            noisy_grad = gep.get_approx_grad(clipped_theta) + residual_grad

            offset = 0
            for n, p in model.named_parameters():
                shape = p.grad.shape
                numel = p.grad.numel()
                p.grad.data = Config.BATCH_SIZE * noisy_grad[offset:offset + numel].view(shape)  # clean_grad[offset:offset + numel].view(shape)
                accumulated_grads[n] += noisy_grad[offset:offset + numel].view(shape)
                offset += numel
            del clean_grad, noisy_grad
        else:  # No GEP
            for i, (n, p) in enumerate(model.named_parameters()):
                accumulated_grads[n] += p.grad.data
        # print('Before Step')
        if Config.TOY_STORY and Config.PLOT_GRADS:
            for ((n, p), (bn, bp)) in zip(model.named_parameters(), bench_model.named_parameters()):
                assert n in grads_plot_list.keys()
                grads_plot_list[n].append(p.grad.data.detach())

                assert bn in bench_grads_plot_list.keys()
                bench_grads_plot_list[bn].append(bp.grad.data.detach())

                # print('@@@ params $$$')
                # print(n, p.data.detach(), bp.data.detach(), p.data.detach() - bp.data.detach())
                #
                # print('### grads &&&')
                # print(n, p.grad.data.detach(), bp.grad.data.detach(), p.grad.data.detach() - bp.grad.data.detach())

        optimizer.step()
        bench_optimizer.step()

        # print('After Step')
        if Config.TOY_STORY and Config.PLOT_GRADS:
            for ((n, p), (bn, bp)) in zip(model.named_parameters(), bench_model.named_parameters()):
                assert n in params_plot_list.keys()
                params_plot_list[n].append(p.data.detach() - params_plot_list[n][-1])
                # params_plot_list[n].append(p.data.detach())

                assert bn in bench_params_plot_list.keys()
                bench_params_plot_list[bn].append(bp.data.detach() - bench_params_plot_list[bn][-1])
                # bench_params_plot_list[bn].append(bp.data.detach())

                # print('@@@ params $$$')
                # print(n, p.data.detach(), bp.data.detach(), p.data.detach() - bp.data.detach())
                #
                # print('### grads &&&')
                # print(n, p.grad.data.detach(), bp.grad.data.detach(), p.grad.data.detach() - bp.grad.data.detach())

        # print('Grads before step')
        # print('1.weight\n',
        #       [(p, bp, p - bp) for (p, bp) in zip(grads_plot_list['1.weight'], bench_grads_plot_list['1.weight'])])
        # print('1.bias\n',
        #       [(p, bp, p - bp) for (p, bp) in zip(grads_plot_list['1.bias'], bench_grads_plot_list['1.bias'])])
        #
        #
        # print('Params after step')
        # print('1.weight\n',
        #       [(p, bp, p - bp) for (p, bp) in zip(params_plot_list['1.weight'], bench_params_plot_list['1.weight'])])
        # print('1.bias\n',
        #       [(p, bp, p - bp) for (p, bp) in zip(params_plot_list['1.bias'], bench_params_plot_list['1.bias'])])

        if not Config.TOY_STORY:
            correct = (predicted == labels).sum().item()
            correct_counter += int(correct)
        else:
            losses.append(float(loss))
            bench_losses.append(float(bench_loss))
        # y_pred += predicted.cpu().tolist()
        # y_labels += labels.cpu().tolist()
        del loss, labels, emg, predicted, batch, outputs, bench_loss, bench_outputs
        torch.cuda.empty_cache()
    # if Config.USE_GEP:
    #     for n, p in model.named_parameters():
    #         p.grad.data = accumulated_grads[n]

    epoch_loss = running_loss / float(counter)
    epoch_acc = 100 * correct_counter / sample_counter

    if Config.PLOT_GRADS:
        zip_params_weight = zip(params_plot_list['2.weight'], bench_params_plot_list['2.weight'])
        # zip_params_bias = zip(params_plot_list['1.bias'], bench_params_plot_list['1.bias'])
        # zip_grads_weight = zip(grads_plot_list['2.weight'], bench_grads_plot_list['2.weight'])
        # zip_grads_bias = zip(grads_plot_list['1.bias'], bench_grads_plot_list['1.bias'])
        zip_losses = zip(losses, bench_losses)
        # print(len(grads_plot_list['1.weight']), len(grads_plot_list['1.bias']))
        fig, axs = plt.subplots(7, 1, figsize=(20, 20))

        # axs[0].plot([float(t[0, 0]) for t in params_plot_list['1.weight']], color='b')
        # axs[1].plot([float(t[1, 0]) for t in params_plot_list['1.weight']], color='r')
        # axs[2].plot([float(t[0]) for t in params_plot_list['1.bias']], color='y')
        # axs[3].plot([float(t[0, 0]) for t in bench_params_plot_list['1.weight']], color='b')
        # axs[4].plot([float(t[1, 0]) for t in bench_params_plot_list['1.weight']], color='r')
        # axs[5].plot([float(t[0]) for t in bench_params_plot_list['1.bias']], color='y')
        # axs[6].plot([float(t[0, 1]) for t in params_plot_list['1.weight']], color='b')
        # axs[7].plot([float(t[1, 1]) for t in params_plot_list['1.weight']], color='r')
        # axs[8].plot([float(t[1]) for t in params_plot_list['1.bias']], color='y')
        # axs[9].plot([float(t[0, 1]) for t in bench_params_plot_list['1.weight']], color='b')
        # axs[10].plot([float(t[1, 1]) for t in bench_params_plot_list['1.weight']], color='r')
        # axs[11].plot([float(t[1]) for t in bench_params_plot_list['1.bias']], color='y')

        axs[0].plot([abs(float(t1[0, 0] - t2[0, 0])) for (t1, t2) in zip_params_weight], color='b')
        axs[1].plot([abs(float(t1[1, 0] - t2[1, 0])) for (t1, t2) in zip_params_weight], color='r')
        # axs[2].plot([abs(float(t1[0] - t2[0])) for (t1, t2) in zip_params_bias], color='y')
        axs[2].plot([abs(float(t1[0, 1] - t2[0, 1])) for (t1, t2) in zip_params_weight], color='b')
        axs[3].plot([abs(float(t1[1, 1] - t2[1, 1])) for (t1, t2) in zip_params_weight], color='r')
        # axs[5].plot([abs(float(t1[1] - t2[1])) for (t1, t2) in zip_params_bias], color='y')
        #
        # axs[6].plot([abs(float(t1[0, 0] - t2[0, 0])) for (t1, t2) in zip_grads_weight], color='b')
        # axs[7].plot([abs(float(t1[1, 0] - t2[1, 0])) for (t1, t2) in zip_grads_weight], color='r')
        # axs[8].plot([abs(float(t1[0] - t2[0])) for (t1, t2) in zip_grads_bias], color='y')
        # axs[9].plot([abs(float(t1[0, 1] - t2[0, 1])) for (t1, t2) in zip_grads_weight], color='b')
        # axs[10].plot([abs(float(t1[1, 1] - t2[1, 1])) for (t1, t2) in zip_grads_weight], color='r')
        # axs[11].plot([abs(float(t1[1] - t2[1])) for (t1, t2) in zip_grads_bias], color='y')

        axs[4].plot(losses)
        axs[5].plot(bench_losses)
        axs[6].plot([abs(l1 - l2) for (l1, l2) in zip_losses])
        # GEP_CLIP0 = 5  # 50
        # GEP_CLIP1 = 1  # 20
        # GEP_POWER_ITER = 1
        # GEP_NUM_GROUPS = 3
        #
        # # TOY STORY
        # TOY_STORY = True
        # PLOT_GRADS = True
        # NUM_OUTPUTS = 200
        # DATA_SCALE = 10.0
        plt.title(f'OUTPUT_DIM = {Config.OUTPUT_DIM},'
                  f' DATA_DIM = {Config.DATA_DIM},  '
                  f'DP_SIGMA = {Config.DP_SIGMA},'
                  f' DATA_SCALE = {Config.DATA_SCALE},'
                  f' CLIP0 = {Config.GEP_CLIP0},'
                  f' CLIP1 = {Config.GEP_CLIP1}')
        plt.show()
        raise Exception
    # train_objects.model = model
    return epoch_loss, epoch_acc, accumulated_grads, model, optimizer


def dp_sgd_fwd_bwd(inputs, labels, train_objects, dp_params, zero_grad_now, grad_step_now):
    model, loader, criterion, optimizer = astuple(train_objects)
    if model.training and zero_grad_now:
        # optimizer.zero_grad() done
        # at first batch or if optimizer.step() done at previous batch
        optimizer.zero_grad()

    outputs = model(inputs.float())
    loss = criterion(outputs, labels.long())
    _, predicted = torch.max(outputs.data, 1)
    if model.training:
        loss.backward()
        if grad_step_now:
            if dp_params is not None:
                add_dp_noise(model, params=dp_params)
            optimizer.step()
    return predicted, float(loss)


def run_single_epoch(train_objects: TrainObjects,
                     train_params: TrainParams,
                     dp_params: DpParams = None,
                     fwd_bwd_fn: Callable[[torch.Tensor, torch.Tensor, TrainObjects, DpParams, bool, bool], (
                             torch.Tensor, int)] = dp_sgd_fwd_bwd) \
        -> (float, float):
    _, batch_size, descent_every, _, _ = astuple(train_params)
    model, loader, criterion, optimizer = astuple(train_objects)

    assert not model.training or optimizer is not None, 'None Optimizer  given at train epoch'
    running_loss, correct_counter, sample_counter, counter = 0, 0, 0, 0
    device = next(model.parameters()).device
    y_pred, y_labels = [], []
    zero_grad_now = True

    for k, batch in enumerate(loader):
        curr_batch_size = batch[1].size(0)
        if curr_batch_size < batch_size:
            continue
        counter += 1

        sample_counter += curr_batch_size
        batch = (t.to(device) for t in batch)
        inputs, labels = batch
        labels = labels_to_consecutive(labels).squeeze()

        grad_step_now = descent_every == 1 or descent_every > 1 and ((k + 1) % descent_every) == 0

        predicted, loss = fwd_bwd_fn(inputs, labels, train_objects, dp_params, zero_grad_now, grad_step_now)
        zero_grad_now = not grad_step_now

        running_loss += loss

        correct = (predicted == labels).sum().item()
        correct_counter += int(correct)

        y_pred += predicted.cpu().tolist()
        y_labels += labels.cpu().tolist()
    loss = running_loss / float(counter)
    acc = 100 * correct_counter / sample_counter
    return loss, acc


def train_model(train_objects: TrainObjects,
                val_objects: TrainObjects,
                test_objects: TrainObjects,
                train_params: TrainParams,
                dp_params=None,
                log2wandb: bool = True,
                show_pbar: bool = True,
                output_fn=lambda s: None):
    num_epochs, _, _, validation_eval_every, test_eval_at_end = astuple(train_params)
    eval_params = TrainParams(epochs=1, batch_size=train_params.batch_size)
    train_loss, train_acc, validation_loss, validation_acc = 0, 0, 0, 0
    eval_num = 0
    range_epochs = range(1, num_epochs + 1)
    epoch_pbar = tqdm(range_epochs) if show_pbar else None
    model = train_objects.model
    model.train()
    for epoch in (epoch_pbar if show_pbar else range_epochs):

        epoch_train_loss, epoch_train_acc = \
            run_single_epoch(train_objects=train_objects,
                             train_params=train_params,
                             dp_params=dp_params)

        if validation_eval_every == 1 or (validation_eval_every > 1 and epoch % (validation_eval_every - 1)) == 0:
            model.eval()
            epoch_validation_loss, epoch_validation_acc = \
                run_single_epoch(train_objects=val_objects,
                                 train_params=eval_params)
            model.train()
            eval_num += 1
            validation_loss += epoch_validation_loss
            validation_acc += epoch_validation_acc
        if show_pbar:
            epoch_pbar.set_description(
                f'epoch {epoch} loss {round(epoch_train_loss, 3)} acc {epoch_train_acc}'
                f' epoch validation loss {round(epoch_validation_loss, 3)}'
                f' epoch validation acc {round(epoch_validation_acc, 3)}')

        train_acc += epoch_train_acc
        train_loss += epoch_train_loss

        if log2wandb:
            wandb.log({'epoch': epoch,
                       'epoch_train_loss': epoch_train_loss,
                       'epoch_train_acc': epoch_train_acc,
                       'epoch_validation_loss': epoch_validation_loss,
                       'epoch_validation_acc': epoch_validation_acc,
                       'train_loss': train_loss / epoch,
                       'train_acc': train_acc / epoch,
                       'validation_loss': validation_loss / eval_num,
                       'validation_acc': validation_acc / eval_num
                       })
    if test_eval_at_end:
        model.eval()
        test_loss, test_acc = run_single_epoch(
            train_objects=test_objects,
            train_params=eval_params)
        output_fn(f'Test Loss {test_loss} Test Acc {test_acc}')
        if log2wandb:
            wandb.log({'test_loss': test_loss, 'test_acc': test_loss})

    return train_loss / num_epochs, train_acc / num_epochs, validation_loss / eval_num, validation_acc / eval_num
