import logging

import torch.nn
import torchvision
from torch import nn
from torchvision.models import ResNet18_Weights
from fed_priv_models.cnn_target import CNNTarget
from common.config import Config
from fed_priv_models.pad_operators import Reshape3Bands, PadBeforeLast, PadLastDimCircular, Squeeze
from fed_priv_models.resnet_cifar import resnet20


def init_model():
    # Define Model
    if Config.TOY_STORY:
        # A simple linear 2-layer Toy model
        model = toy_model()
    elif Config.CIFAR_DATA:
        # model = CNNTarget(in_channels=3, n_kernels=16, embedding_dim=10 if Config.CIFAR10_DATA else 100)
        # model = simple_mlp_cls()
        model = resnet20()
        # model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        # Model3d
        model = model3d()
    logging.info(f'Model {model} constructed.  Init model weights.')
    # Init model weights
    for m in model.modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
    return model


def model3d():
    model = torch.nn.Sequential(torch.nn.Sequential(

        Reshape3Bands(window_size=Config.WINDOW_SIZE, W=3, H=8),
        PadBeforeLast(),
        PadLastDimCircular(window_size=Config.WINDOW_SIZE, W=3),

        # Conv3DBlock
        torch.nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
        torch.nn.AvgPool3d(kernel_size=(1, 3, 1), stride=(1, 3, 1), padding=0),
        Squeeze(),
        # torch.nn.GroupNorm(4, 32, eps=1e-05, affine=True),
        torch.nn.ReLU(inplace=False),

        # Conv2DBlock
        torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2)),
        torch.nn.AvgPool2d(kernel_size=(3, 3), stride=(3, 3), padding=0),
        Squeeze(),
        # torch.nn.GroupNorm(4, 64, eps=1e-05, affine=True),
        torch.nn.ReLU(inplace=False),

        # Conv1DBlock
        torch.nn.Conv1d(64, 128, kernel_size=(3,), stride=(2,)),
        torch.nn.AvgPool1d(kernel_size=(3,), stride=(3,), padding=(0,)),
        # torch.nn.GroupNorm(4, 128, eps=1e-05, affine=True),
        torch.nn.ReLU(inplace=False),

        # Conv1DBlock
        torch.nn.Conv1d(128, 256, kernel_size=(3,), stride=(2,)),
        torch.nn.AvgPool1d(kernel_size=(3,), stride=(3,), padding=(0,)),
        # torch.nn.GroupNorm(4, 128, eps=1e-05, affine=True),
        torch.nn.ReLU(inplace=False),

        # FlattenToLinear(),
        torch.nn.Flatten(start_dim=1, end_dim=-1),
        # DenseBlock
        torch.nn.Linear(in_features=256, out_features=256, bias=True),
        torch.nn.ReLU(inplace=False),

        # DenseBlock
        torch.nn.Linear(in_features=256, out_features=128, bias=True),
        torch.nn.ReLU(inplace=False),

        # DenseBlock
        torch.nn.Linear(in_features=128, out_features=64, bias=True),
        torch.nn.ReLU(inplace=False),

        # output layer
        torch.nn.Linear(in_features=64, out_features=7, bias=True)
    ))
    return model


def simple_mlp_cls():
    model = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=1, end_dim=-1),
        torch.nn.Linear(in_features=32 * 32 * 3, out_features=Config.HIDDEN_DIM, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=Config.HIDDEN_DIM, out_features=10, bias=True),
        torch.nn.Softmax(dim=-1))
    return model


def toy_model():
    model = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=1, end_dim=-1),
        torch.nn.Linear(in_features=Config.DATA_DIM, out_features=Config.HIDDEN_DIM, bias=True),
        torch.nn.Linear(in_features=Config.HIDDEN_DIM, out_features=Config.OUTPUT_DIM, bias=True))
    return model
