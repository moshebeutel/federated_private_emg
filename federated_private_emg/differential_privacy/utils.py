from __future__ import annotations

from dataclasses import astuple
from typing import Callable

from backpack import backpack, extend
from backpack.extensions import BatchGrad, BatchL2Grad
import torch
import wandb
from collections.abc import Callable
from common.config import Config
from common.utils import calc_grad_norm
from differential_privacy.params import DpParams
from fed_priv_models.gep import GEP


def add_dp_noise(model,
                 params: DpParams = DpParams(dp_lot=Config.LOT_SIZE_IN_BATCHES * Config.BATCH_SIZE,
                                             dp_sigma=Config.DP_SIGMA,
                                             dp_C=Config.DP_C)):
    dp_lot, dp_sigma, dp_C = astuple(params)
    grad_norm = calc_grad_norm(model)

    for p in model.parameters():
        # Clip gradients
        p.grad /= max(1, grad_norm / dp_C)
        # Add DP noise to gradients
        noise = torch.normal(mean=0, std=dp_sigma * dp_C, size=p.grad.size(), device=p.device)
        # noise = torch.randn_like(p.grad) * dp_sigma * dp_C
        p.grad += noise
        p.grad /= dp_lot  # Averaging over lot


def add_dp_noise_using_per_sample(model,
                                  params: DpParams = DpParams(dp_lot=Config.LOT_SIZE_IN_BATCHES * Config.BATCH_SIZE,
                                                              dp_sigma=Config.DP_SIGMA, dp_C=Config.DP_C)):
    dp_lot, dp_sigma, dp_C = astuple(params)

    for name, param in model.named_parameters():
        # Clip gradients
        param.grad_batch /= max(1, param.batch_l2 / dp_C)
        # Add DP noise to gradients
        noise = torch.normal(mean=0, std=dp_sigma * dp_C, size=param.grad.size(), device=param.device)
        param.grad += noise
        param.grad /= dp_lot  # Averaging over lot


def per_sample_gradient_fwd_bwd(model: torch.nn.Module,
                                loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                batch_input: torch.Tensor,
                                batch_labels: torch.Tensor,
                                perturb_fn: Callable[[torch.Tensor], torch.Tensor]):
    model = extend(model)
    loss_fn = extend(loss_fn)
    output = model(batch_input)

    loss = loss_fn(output, batch_labels)

    with backpack(BatchGrad()):
        loss.backward()

    for name, param in model.named_parameters():
        print(name)
        print(".grad.shape:             ", param.grad.shape)
        print(".grad_batch.shape:       ", param.grad_batch.shape)
        param.grad = perturb_fn(param.grad_batch)


def per_sample_gradient_fwd_bwd(inputs, labels, train_objects, dp_params, zero_grad_now, grad_step_now):
    model, loader, criterion, optimizer = astuple(train_objects)
    if model.training and zero_grad_now:
        # optimizer.zero_grad() done
        # at first batch or if optimizer.step() done at previous batch
        optimizer.zero_grad()

    model = extend(model)
    criterion = extend(criterion)
    outputs = model(inputs.float())
    loss = criterion(outputs, labels.long())
    _, predicted = torch.max(outputs.data, 1)
    if model.training:
        with backpack(BatchGrad()):
            with backpack(BatchL2Grad()):
                loss.backward()
        if grad_step_now:
            if dp_params is not None:
                add_dp_noise_using_per_sample(model, params=dp_params)
            optimizer.step()
    return predicted, float(loss)


def attach_gep(net: torch.nn.Module, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], num_bases: int,
               batch_size: int, clip0: float, clip1: float, power_iter: int,
               num_groups: int, public_inputs: torch.Tensor, public_targets: torch.Tensor):
    device = next(net.parameters()).device
    # print('\n==> Creating GEP class instance')
    gep = GEP(num_bases, batch_size, clip0, clip1, power_iter).cuda()
    ## attach auxiliary data to GEP instance
    public_inputs, public_targets = public_inputs.to(device), public_targets.to(device)
    gep.public_inputs = public_inputs
    gep.public_targets = public_targets
    gep.loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(public_inputs, public_targets),
            shuffle=False,
            batch_size=batch_size,
            num_workers=Config.NUM_WORKERS
        )

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
