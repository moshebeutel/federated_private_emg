from __future__ import annotations

from dataclasses import astuple

import torch
import wandb

from common.config import Config
from common.utils import calc_grad_norm
from differential_privacy.params import DpParams


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
