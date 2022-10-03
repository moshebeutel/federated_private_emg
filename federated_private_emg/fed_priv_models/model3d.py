import logging

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils import config_logger, LOG_FOLDER


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


class Model3d(nn.Module):
    def __init__(self, number_of_classes,
                 window_size=1280,
                 depthwise_multiplier=32,
                 W=3,
                 H=8,
                 use_batch_norm=False,
                 use_dropout=False):
        super(Model3d, self).__init__()
        self.logger = config_logger(f'{Model3d}_logger', level=logging.DEBUG, log_folder=LOG_FOLDER)
        self._window = window_size > 3
        w_ker_siz = 3 if self._window else 1
        hw_ker_siz = 1 if self._window else 0
        self._reshape = lambda x: x.reshape(x.shape[0], 1, window_size, W, H)
        self._pad_before_last_dim_constant = lambda x: \
            F.pad(x, pad=(0, 0, 1, 1, hw_ker_siz, hw_ker_siz), mode='constant')
        self._pad_last_dim_circular = lambda x: \
            torch.concat([x[:, :, :, :, -1].reshape(-1, 1, window_size + 2 * hw_ker_siz, W + 2, 1),
                          x, x[:, :, :, :, 0].reshape(-1, 1, window_size + 2 * hw_ker_siz, W + 2, 1)], axis=4)

        # Downsample 3 stripes to 1
        self._conv_block1 = Conv3DBlock(depthwise_multiplier, depthwise_multiplier,
                                        kernel_size=(w_ker_siz, 3, 3), pool_kernel_size=(1, 3, 1),
                                        use_batchnorm=True, use_dropout=True)

        # self._conv1 = nn.Conv3d(1, depthwise_multiplier, groups=1, kernel_size=(w_ker_siz, 3, 3))
        # self._pool1 = nn.AvgPool3d(kernel_size=(1, 3, 1))
        # self._batch_norm1 = nn.BatchNorm3d(depthwise_multiplier) if use_batch_norm \
        #     else torch.nn.identity(depthwise_multiplier)
        # # self._prelu1 = nn.ReLU(depthwise_multiplier)
        # self._dropout1 = nn.Dropout3d(.5) if use_dropout \
        #     else torch.nn.identity(depthwise_multiplier)

        # 1st Downsample time dimension
        if self._window:
            self._conv_block2 = Conv2DBlock(depthwise_multiplier, 2 * depthwise_multiplier,
                                            kernel_size=(w_ker_siz, 3),
                                            stride=(2, 2),
                                            pool_kernel_size=(3, 3), use_batchnorm=True, use_dropout=True)

            # self._conv2 = nn.Conv2d(depthwise_multiplier, 2 * depthwise_multiplier, kernel_size=(w_ker_siz, 3),
            #                         stride=(2, 2))
            # self._pool2 = nn.AvgPool2d(kernel_size=(3, 3))
            # self._batch_norm2 = nn.BatchNorm2d(2 * depthwise_multiplier) if use_batch_norm \
            #     else torch.nn.identity(2 * depthwise_multiplier)
            # # self._prelu2 = nn.ReLU(2 * depthwise_multiplier)
            # self._dropout2 = nn.Dropout2d(.5) if use_dropout else torch.nn.identity(2 * depthwise_multiplier)

            # = nn.Conv2d(2*depthwise_multiplier, 4 * depthwise_multiplier, kernel_size=(w_ker_siz, 1),
            #                        stride=(2, 1))

            self._conv_block3 = Conv1DBlock(2 * depthwise_multiplier, 4 * depthwise_multiplier,
                                            kernel_size=w_ker_siz,
                                            stride=2,
                                            pool_kernel_size=3, use_batchnorm=True, use_dropout=True)
            # self._conv3 = nn.Conv1d(2 * depthwise_multiplier, 4 * depthwise_multiplier, kernel_size=w_ker_siz, stride=2)
            # self._pool3 = nn.AvgPool1d(kernel_size=3)
            # self._batch_norm3 = nn.BatchNorm1d(4 * depthwise_multiplier) if use_batch_norm \
            #     else torch.nn.identity(4 * depthwise_multiplier)
            # # self._prelu3 = nn.ReLU(4 * depthwise_multiplier)
            # self._dropout3 = nn.Dropout(.5) if use_dropout else torch.nn.identity(4 * depthwise_multiplier)

            self._conv_block4 = Conv1DBlock(4 * depthwise_multiplier, 8 * depthwise_multiplier,
                                            kernel_size=w_ker_siz,
                                            stride=2,
                                            pool_kernel_size=3, use_batchnorm=True, use_dropout=True)
            # self._conv4 = nn.Conv1d(4 * depthwise_multiplier, 8 * depthwise_multiplier, kernel_size=w_ker_siz, stride=2)
            # self._pool4 = nn.AvgPool1d(kernel_size=3)
            # self._batch_norm4 = nn.BatchNorm1d(8 * depthwise_multiplier) if use_batch_norm \
            #     else torch.nn.identity(8 * depthwise_multiplier)
            # # self._prelu4 = nn.ReLU(8 * depthwise_multiplier)
            # self._dropout4 = nn.Dropout(.5) if use_dropout else torch.nn.identity(8 * depthwise_multiplier)

        self.flatten = lambda x: x.view(-1, 8 * depthwise_multiplier)

        self._dense_block1 = DenseBlock(8 * depthwise_multiplier, 8 * depthwise_multiplier,
                                        use_batchnorm=True, use_dropout=True)
        # self._fc1 = nn.Linear(W*H, 8*depthwise_multiplier)
        # self._fc1 = nn.Linear(8 * depthwise_multiplier, 8 * depthwise_multiplier)
        # self._batch_norm5 = nn.BatchNorm1d(8*depthwise_multiplier)
        # self._prelu5 = nn.ReLU(8*depthwise_multiplier)
        # self._dropout5 = nn.Dropout(.5)
        self._dense_block2 = DenseBlock(8 * depthwise_multiplier, 4 * depthwise_multiplier,
                                        use_batchnorm=True, use_dropout=True)
        # self._fc2 = nn.Linear(8 * depthwise_multiplier, 4 * depthwise_multiplier)
        # self._batch_norm6 = nn.BatchNorm1d(4 * depthwise_multiplier)
        # self._prelu6 = nn.ReLU(4 * depthwise_multiplier)
        # self._dropout6 = nn.Dropout(.5)

        self._dense_block3 = DenseBlock(4 * depthwise_multiplier, 2 * depthwise_multiplier,
                                        use_batchnorm=True, use_dropout=True)
        # self._fc3 = nn.Linear(4 * depthwise_multiplier, 2 * depthwise_multiplier)
        # self._batch_norm6 = nn.BatchNorm1d(2 * depthwise_multiplier)
        # self._prelu7 = nn.ReLU(2*depthwise_multiplier)
        # self._dropout7 = nn.Dropout(.5)

        self._output = nn.Linear(2 * depthwise_multiplier, number_of_classes)

        self.initialize_weights()

        self.logger.info(str(self))

        self.logger.info(f"Number Parameters: {self.get_n_params()}")

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def init_weights(self):
        for m in self.modules():
            torch.nn.init.kaiming_normal(m.weight)
            m.bias.data.zero_()

    def initialize_weights(self):
        for m in self.modules():

            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        self.logger.debug(f'input {x.shape}')
        x = self._reshape(x)
        self.logger.debug(f'reshape {x.shape}')
        x = self._pad_before_last_dim_constant(x)
        self.logger.debug(f'_pad_before_last_dim_constant {x.shape}')
        x = self._pad_last_dim_circular(x)
        self.logger.debug(f'pad_last_dim {x.shape}')

        conv1 = self._conv_block1(x)

        # conv1 = F.relu(self._conv1(x))
        # conv1 = self._prelu1(self._conv1(x))
        # conv1 = self._dropout1(self._prelu1(self._conv1(x)))
        # conv1 = self._dropout1(self._prelu1(self._batch_norm1(self._conv1(x))))
        # self.logger.debug(f'conv1 {conv1.shape}')
        # pool1 = self._pool1(conv1)
        # self.logger.debug(f'pool1 {pool1.shape}')

        if self._window:

            conv2 = self._conv_block2(conv1.squeeze(dim=3))

            # conv2 = F.relu(self._conv2(pool1.squeeze(dim=3)))
            # conv2 = self._dropout2(self._prelu2(self._batch_norm2(self._conv2(pool1.squeeze()))))
            # self.logger.debug(f'conv2 {conv2.shape}')
            # pool2 = self._pool2(conv2)
            # self.logger.debug(f'pool2 {pool2.shape}')

            conv3 = self._conv_block3(conv2.squeeze(dim=3))

            # conv3 = self._conv3(pool2.squeeze(dim=3))
            # self.logger.debug(f'conv3 {conv3.shape}')
            # conv3 = self._batch_norm3(conv3)
            # self.logger.debug(f'batchnorm {conv3.shape}')
            # conv3 = F.relu(conv3)
            # conv3 = self._dropout3(self._prelu3(conv3))
            # self.logger.debug(f'relu3 {conv3.shape}')
            # self.logger.debug(f'dropout3 {conv3.shape}')
            # pool3 = self._pool3(conv3)
            # self.logger.debug(f'pool3 {pool3.shape}')

            conv4 = self._conv_block4(conv3)

            # conv4 = self._conv4(pool3)
            # self.logger.debug(f'conv4 {conv4.shape}')
            # # conv4 = self._batch_norm4(conv4)
            # # self.logger.debug(f'batchnorm {conv4.shape}')
            # conv4 = F.relu(conv4)
            # # conv4 = self._dropout4(self._prelu4(conv4))
            # self.logger.debug(f'relu4 {conv4.shape}')
            # # self.logger.debug(f'dropout4 {conv4.shape}')
            # pool4 = self._pool3(conv4)
            # self.logger.debug(f'pool4 {pool4.shape}')
        else:
            self.logger.debug('no time window')
            # pool4 = pool1
            conv4 = conv1
        flatten_tensor = self.flatten(conv4)
        # flatten_tensor = self.flatten(pool4)
        # self.logger.debug(f'flatten_tensor {flatten_tensor.shape}')
        # if flatten_tensor.size(0) != x.size(0):
        #     print()

        fc1 = self._dense_block1(flatten_tensor)
        # fc1 = F.relu(self._fc1(flatten_tensor))
        # fc1 = self._prelu5(self._fc1(x))
        # fc1 = self._prelu5(self._fc1(flatten_tensor))
        # fc1 = self._dropout5(self._prelu5(self._fc1(flatten_tensor)))
        # fc1 = self._dropout5(self._prelu5(self._batch_norm5(self._fc1(flatten_tensor))))
        # self.logger.debug(f'fc1 {fc1.shape}')

        fc2 = self._dense_block2(fc1)
        # fc2 = F.relu(self._fc2(fc1))
        # fc2 = self._prelu6(self._fc2(fc1))
        # fc2 = self._dropout6(self._prelu6(self._fc2(fc1)))
        # fc2 = self._dropout6(self._prelu6(self._batch_norm6(self._fc2(fc1))))
        # self.logger.debug(f'fc2 {fc2.shape}')

        fc3 = self._dense_block3(fc2)
        # fc3 = F.relu(self._fc3(fc2))
        # fc3 = self._prelu7(self._fc3(fc2))
        # fc3 = self._dropout6(self._prelu7(self._fc3(fc2)))
        # fc3 = self._dropout7(self._prelu7(self._batch_norm7(self._fc3(fc2))))
        # self.logger.debug(f'fc3 {fc3.shape}')
        output = self._output(fc3)
        self.logger.debug(f'logits {output.shape}')
        # output = F.softmax(output, dim=1)
        # self.logger.debug(f'softmax {output.shape}')
        return output


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(message)s')
    window = 259
    model = Model3d(number_of_classes=64, window_size=window, depthwise_multiplier=32)

    batch = 200

    channels = 1
    W = 3
    H = 8

    inp = torch.arange(batch * window * channels * W * H).float().reshape(batch, window, channels, W * H)

    l = model(inp)
