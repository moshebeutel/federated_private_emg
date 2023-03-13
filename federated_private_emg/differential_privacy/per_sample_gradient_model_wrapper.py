import torch.nn
from torch import nn

from backpack import extend


class PsgModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super(PsgModelWrapper, self).__init__()
        self._underlying_model = model
        self.model = extend(model)

    def forward(self, x):
        return self.model(x)


class PsgLossFnWrapper:
    def __init__(self, loss_fn=torch.nn.CrossEntropyLoss):
        self.wrapped_loss_fn = loss_fn
        self.extended_loss_fn = extend(loss_fn)
