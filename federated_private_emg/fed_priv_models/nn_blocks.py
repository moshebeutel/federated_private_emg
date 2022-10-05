import torch
from torch import nn as nn
from torch.nn import functional as F


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 pool_kernel_size,
                 use_batchnorm=False,
                 use_dropout=False):
        super(Conv3DBlock, self).__init__()
        self._conv = nn.Conv3d(1, in_channels, groups=1, kernel_size=kernel_size)
        self._pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self._batch_norm = nn.BatchNorm3d(out_channels) if use_batchnorm \
            else torch.nn.Identity(out_channels)
        self._relu = nn.ReLU(out_channels)
        self._dropout = nn.Dropout3d(.5) if use_dropout \
            else torch.nn.Identity(out_channels)

    def forward(self, x):
        # return self._dropout(self._relu(self._batch_norm(self._conv(x))))
        return F.relu(self._dropout(self._batch_norm(self._pool(self._conv(x)))))


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size, stride,
                 pool_kernel_size,
                 use_batchnorm=False,
                 use_dropout=False):
        super(Conv2DBlock, self).__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self._pool = nn.AvgPool2d(kernel_size=pool_kernel_size)
        self._batch_norm = nn.BatchNorm2d(out_channels) if use_batchnorm \
            else torch.nn.Identity(out_channels)
        self._relu = nn.ReLU(out_channels)
        self._dropout = nn.Dropout2d(.5) if use_dropout \
            else torch.nn.Identity(out_channels)

    def forward(self, x):
        # return self._dropout(self._relu(self._batch_norm(self._conv(x))))
        return F.relu(self._dropout(self._batch_norm(self._pool(self._conv(x)))))


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size, stride,
                 pool_kernel_size,
                 use_batchnorm=False,
                 use_dropout=False):
        super(Conv1DBlock, self).__init__()
        self._conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self._pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        self._batch_norm = nn.BatchNorm1d(out_channels) if use_batchnorm \
            else torch.nn.Identity(out_channels)
        self._relu = nn.ReLU(out_channels)
        self._dropout = nn.Dropout(.5) if use_dropout \
            else torch.nn.Identity(out_channels)

    def forward(self, x):
        # return self._dropout(self._relu(self._batch_norm(self._conv(x))))
        return F.relu(self._dropout(self._batch_norm(self._pool(self._conv(x)))))


class DenseBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 use_batchnorm=False,
                 use_dropout=False):
        super(DenseBlock, self).__init__()
        self._fc = nn.Linear(in_channels, out_channels)
        self._batch_norm = nn.BatchNorm1d(out_channels) if use_batchnorm \
            else torch.nn.Identity(out_channels)
        self._relu = nn.ReLU(out_channels)
        self._dropout = nn.Dropout(.5) if use_dropout else torch.nn.Identity(out_channels)

    def forward(self, x):
        # return self._dropout(self._relu(self._batch_norm(self._fc(x))))
        return F.relu(self._dropout(self._batch_norm(self._fc(x))))
