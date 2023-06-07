from __future__ import annotations

import gc
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
from differential_privacy.utils import per_sample_gradient_fwd_bwd
from differential_privacy.accountant_utils import add_dp_noise
from fed_priv_models.gep import GEP
from train.params import TrainParams
from train.train_objects import TrainObjects
from collections.abc import Callable
from common.config import Config
import matplotlib.pyplot as plt


def sgd_dp_batch(model, batchsize):
    device = next(model.parameters()).device
    grad_norm_list = torch.zeros(batchsize).to(device)
    for p in model.parameters():
        flat_g = p.grad_batch.reshape(batchsize, -1)
        current_norm_list = torch.norm(flat_g, dim=1)
        grad_norm_list += torch.square(current_norm_list)
    grad_norm_list = torch.sqrt(grad_norm_list) / Config.DP_C
    grad_norm_list = torch.clip(grad_norm_list, min=1)

    for p in model.parameters():
        flat_g = p.grad_batch.reshape(batchsize, -1)
        flat_g = torch.div(flat_g, grad_norm_list.reshape(-1, 1))
        p.grad_batch = flat_g.reshape(batchsize, *p.shape)
        p.grad = torch.mean(p.grad_batch, dim=0)
        del p.grad_batch
        # Add DP noise to gradients
        noise = torch.normal(mean=0, std=Config.DP_SIGMA * Config.DP_C, size=p.grad.size(), device=p.device) / batchsize
        p.grad += noise
    del grad_norm_list


def run_single_epoch_keep_grads(model, optimizer, loader, criterion,
                                batch_size: int, gep: GEP = None,
                                gp=None,
                                use_dp_noise: bool = False) -> (float, float):
    assert model.training, 'Model not in train mode'
    running_loss, correct_counter, sample_counter = 0., 0, 0
    device = next(model.parameters()).device

    bench_model = copy.deepcopy(model)

    if Config.INTERNAL_BENCHMARK:
        bench_model.train()

    # Init grads accumulator
    # accumulated_grads = OrderedDict()
    for ((n, p), (bn, bp)) in zip(model.named_parameters(), bench_model.named_parameters()):
        # accumulated_grads[n] = torch.zeros_like(p.data)
        bp.data = copy.deepcopy(p.data)
        p.grad = torch.zeros_like(p.data)
        bp.grad = torch.zeros_like(bp.data)

    if Config.INTERNAL_BENCHMARK:
        bench_optimizer = torch.optim.SGD(bench_model.parameters(), lr=Config.LEARNING_RATE,
                                          weight_decay=Config.WEIGHT_DECAY, momentum=0.9)

    if Config.TOY_STORY and Config.PLOT_GRADS and Config.INTERNAL_BENCHMARK:
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
    for epoch in range(Config.NUM_INTERNAL_EPOCHS):
        # print('internal epoch', epoch)
        for k, batch in enumerate(loader):

            curr_batch_size = batch[1].size(0)
            # if curr_batch_size < batch_size:
            #     continue
            # counter += 1

            sample_counter += curr_batch_size
            batch = (t.to(device) for t in batch)
            emg, labels = batch

            labels = labels if Config.TOY_STORY or Config.CIFAR10_DATA else labels_to_consecutive(labels).long().squeeze()

            optimizer.zero_grad()
            outputs = model(emg.float())

            if Config.USE_GP:
                X = torch.cat((X, outputs), dim=0) if k > 0 else outputs
                Y = torch.cat((Y, labels), dim=0) if k > 0 else labels
            else:
                # print('outputs.shape', outputs.shape)
                loss = criterion(outputs, labels)
                # print('loss', loss)
                running_loss += float(loss)
                if Config.USE_GEP and Config.INTERNAL_BENCHMARK:
                    with backpack(BatchGrad()):
                        loss.backward()
                else:
                    loss.backward()

                # print('running_loss', running_loss)
                if Config.TOY_STORY:
                    pass
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    # print('predicted', predicted)
                    correct = (predicted == labels).sum().item()
                    # print('correct', correct, ' out of ', predicted.shape)
                    correct_counter += int(correct)
                    del predicted
                del outputs


                # print('after backward')

                if Config.INTERNAL_BENCHMARK:
                    bench_optimizer.zero_grad()
                    bench_outputs = bench_model(emg.float())
                    bench_loss = criterion(bench_outputs, labels)
                    bench_loss.backward()
                    del bench_outputs

                if Config.USE_GEP and Config.INTERNAL_BENCHMARK:
                    gep_batch(None, gep, model)
                    # gep_batch(accumulated_grads, gep, model)
                else:  # No internal GEP
                    lr = optimizer.param_groups[0]['lr']

                if Config.TOY_STORY and Config.PLOT_GRADS:
                    for ((n, p), (bn, bp)) in zip(model.named_parameters(), bench_model.named_parameters()):
                        assert n in grads_plot_list.keys()
                        grads_plot_list[n].append(p.grad.data.detach())

                        assert bn in bench_grads_plot_list.keys()
                        bench_grads_plot_list[bn].append(bp.grad.data.detach())

                optimizer.step()

                if Config.INTERNAL_BENCHMARK:
                    bench_optimizer.step()

                if Config.TOY_STORY and Config.PLOT_GRADS:
                    losses.append(float(loss))

                    if Config.INTERNAL_BENCHMARK:
                        bench_losses.append(float(bench_loss))
                        del bench_loss
                    if Config.PLOT_GRADS:
                        for ((n, p), (bn, bp)) in zip(model.named_parameters(), bench_model.named_parameters()):
                            assert n in params_plot_list.keys()
                            params_plot_list[n].append(p.data.detach() - params_plot_list[n][-1])
                            # params_plot_list[n].append(p.data.detach())

                            assert bn in bench_params_plot_list.keys()
                            bench_params_plot_list[bn].append(bp.data.detach() - bench_params_plot_list[bn][-1])
                            # bench_params_plot_list[bn].append(bp.data.detach())

            del labels, emg, batch

            if not Config.USE_GP:
                del loss

            # Release GPU and CPU memory - or at least try to ;)
            torch.cuda.empty_cache()
            gc.collect()

    if Config.USE_GP:
        assert gp is not None, f'Config.USE_GP={Config.USE_GP} but gp is None'
        loss_from_gp = gp(X, Y)
        if hasattr(loss_from_gp, 'backward'):
            loss_from_gp.backward()
            running_loss += float(loss_from_gp)
            optimizer.step()
        else:
            assert loss_from_gp == 0, f'loss is not a tensor if there was no relevant classes'

        del X, Y, loss_from_gp
        # Release GPU and CPU memory - or at least try to ;)
        torch.cuda.empty_cache()
        gc.collect()

    epoch_loss = running_loss / float(sample_counter)
    epoch_acc = 100 * correct_counter / sample_counter

    if Config.PLOT_GRADS:
        zip_params_weight = zip(params_plot_list['2.weight'], bench_params_plot_list['2.weight'])
        # zip_params_bias = zip(params_plot_list['2.bias'], bench_params_plot_list['1.bias'])
        # zip_grads_weight = zip(grads_plot_list['2.weight'], bench_grads_plot_list['2.weight'])
        # zip_grads_bias = zip(grads_plot_list['2.bias'], bench_grads_plot_list['2.bias'])
        zip_losses = zip(losses, bench_losses)
        # print(len(grads_plot_list['1.weight']), len(grads_plot_list['1.bias']))
        fig, axs = plt.subplots(7, 1, figsize=(20, 20))

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

        plt.title(f'OUTPUT_DIM = {Config.OUTPUT_DIM},'
                  f' DATA_DIM = {Config.DATA_DIM},  '
                  f'DP_SIGMA = {Config.DP_SIGMA},'
                  f' DATA_SCALE = {Config.DATA_SCALE},'
                  f' CLIP0 = {Config.GEP_CLIP0},'
                  f' CLIP1 = {Config.GEP_CLIP1}')
        plt.show()
        raise Exception

    # GEP - perturb local gradients before returning to server
    if not Config.INTERNAL_BENCHMARK:
        with torch.no_grad():
            for ((n, p), (bn, bp)) in zip(model.named_parameters(), bench_model.named_parameters()):
                diff = p.data - bp.data
                bp.grad.data = -diff.unsqueeze(dim=0)
                # bp.grad_batch = -diff.unsqueeze(dim=0)
                # print(bn, bp.grad_batch, bp.grad)

            # for bn, bp in bench_model.named_parameters():
            #     print(bn, bp.grad)
    return epoch_loss, epoch_acc, None, bench_model, optimizer
    # return epoch_loss, epoch_acc, accumulated_grads, bench_model, optimizer


# @profile
def gep_batch(accumulated_grads, gep, model, batchsize):
    assert gep is not None, f'Config.USE_GEP={Config.USE_GEP} but gep object not provided. gep={gep}'
    batch_grad_list = []
    for p in model.parameters():
        if hasattr(p, 'grad_batch'):
            print('p.grad_batch.shape', p.grad_batch.shape)
            batch_grad_list.append(p.grad_batch.reshape(p.grad_batch.shape[0], -1))
            # del p.grad_batch

    print('flatten_tensor(batch_grad_list).shape', flatten_tensor(batch_grad_list).shape)
    clipped_theta, residual_grad, target_grad = gep(flatten_tensor(batch_grad_list))
    del batch_grad_list
    gc.collect()
    # clean_grad = gep.get_approx_grad(clipped_theta) + residual_grad
    if Config.ADD_DP_NOISE:
        # Perturbation
        theta_noise = torch.normal(0, Config.GEP_SIGMA0 * Config.GEP_CLIP0,
                                   size=clipped_theta.shape,
                                   device=clipped_theta.device) / batchsize
        grad_noise = torch.normal(0, Config.GEP_SIGMA1 * Config.GEP_CLIP1,
                                  size=residual_grad.shape,
                                  device=residual_grad.device) / batchsize

        # print('theta_noise', torch.linalg.norm(theta_noise))
        # print('grad_noise', torch.linalg.norm(grad_noise))
        clipped_theta += theta_noise
        residual_grad += grad_noise
        del theta_noise, grad_noise
    # print('clipped_theta', clipped_theta, torch.linalg.norm(clipped_theta))
    # print('residual_grad', residual_grad, torch.linalg.norm(residual_grad))
    # print('target_grad', target_grad, torch.linalg.norm(target_grad))
    # # print('gep.get_approx_grad(clipped_theta)', gep.get_approx_grad(clipped_theta))
    # print('clean_grad', clean_grad, torch.linalg.norm(clean_grad))
    noisy_grad = gep.get_approx_grad(clipped_theta)
    if Config.GEP_USE_RESIDUAL:
        noisy_grad += residual_grad
    # print('noisy_grad', noisy_grad, torch.linalg.norm(noisy_grad))

    offset = 0
    for n, p in model.named_parameters():
        shape = p.grad.shape
        numel = p.grad.numel()

        # The following are for internal benchmark. Otherwise,
        # Internal train is pure sgd (No projection, No clip, No noise) -
        # GEP is done on accumulated grads before publishing

        p.grad.data = noisy_grad[offset:offset + numel].view(shape)
        # p.grad.data *= batchsize
        # p.grad.data *= Config.BATCH_SIZE
        # p.grad.data = Config.BATCH_SIZE * clean_grad[offset:offset + numel].view(shape)
        if accumulated_grads is not None:
            accumulated_grads[n] += (Config.BATCH_SIZE * noisy_grad[offset:offset + numel].view(shape))
        offset += numel
    del noisy_grad
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dp_sgd_fwd_bwd(inputs, labels, model, criterion, optimizer, dp_params, zero_grad_now, grad_step_now):
    # model, loader, criterion, optimizer = astuple(train_objects)
    if model.training and zero_grad_now:
        # optimizer.zero_grad() done
        # at first batch or if optimizer.step() done at previous batch
        optimizer.zero_grad()

    outputs = model(inputs.float())
    loss = criterion(outputs, labels.long())
    _, predicted = (None, 0) if Config.TOY_STORY else torch.max(outputs.data, 1)
    if model.training:
        loss.backward()
        if grad_step_now:
            if dp_params is not None:
                add_dp_noise(model, params=dp_params)
            optimizer.step()
    floss = float(loss)
    del loss, outputs
    gc.collect()
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    return predicted, float(floss)


def run_single_epoch(model, loader, criterion,
                     # train_objects: TrainObjects,
                     train_params: TrainParams,
                     optimizer=None,
                     dp_params: DpParams = None,
                     fwd_bwd_fn: Callable[[torch.Tensor, torch.Tensor, TrainObjects, DpParams, bool, bool], (
                             torch.Tensor, int)] = dp_sgd_fwd_bwd) \
        -> (float, float):
    _, batch_size, descent_every, _, _ = astuple(train_params)
    # model, loader, criterion, optimizer = astuple(train_objects)

    assert not model.training or optimizer is not None, 'None Optimizer  given at train epoch'
    running_loss, correct_counter, sample_counter, counter = 0, 0, 0, 0
    device = next(model.parameters()).device
    y_pred, y_labels = [], []
    zero_grad_now = True

    for k, batch in enumerate(loader):
        curr_batch_size = batch[1].size(0)

        if batch_size > 0 and curr_batch_size < batch_size:
            continue
        counter += 1

        sample_counter += curr_batch_size
        batch = (t.to(device) for t in batch)
        inputs, labels = batch

        labels = labels if Config.TOY_STORY or Config.CIFAR10_DATA else labels_to_consecutive(labels).squeeze()

        grad_step_now = descent_every == 1 or descent_every > 1 and ((k + 1) % descent_every) == 0

        predicted, loss = fwd_bwd_fn(inputs=inputs, labels=labels, model=model, criterion=criterion,
                                     optimizer=optimizer,
                                     dp_params=dp_params, zero_grad_now=zero_grad_now, grad_step_now=grad_step_now)
        zero_grad_now = not grad_step_now

        running_loss += loss

        correct = (predicted == labels).sum().item()
        correct_counter += int(correct)

        y_pred += ([] if Config.TOY_STORY else predicted.cpu().tolist())
        y_labels += labels.cpu().tolist()
        del labels, inputs, batch

    # epoch_loss = running_loss / float(counter)
    epoch_loss = running_loss / float(sample_counter)
    epoch_acc = 100 * correct_counter / float(sample_counter)
    return epoch_loss, epoch_acc


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
