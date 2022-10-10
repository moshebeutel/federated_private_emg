from dataclasses import dataclass

import torch.nn


@dataclass
class TrainObjects:
    model: torch.nn.Module
    loader: torch.utils.data.DataLoader
    criterion: torch.nn.CrossEntropyLoss
    optimizer: torch.optim.Optimizer

    def __init__(self, model, loader, criterion=torch.nn.CrossEntropyLoss(), optimizer=None):
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.optimizer = optimizer
