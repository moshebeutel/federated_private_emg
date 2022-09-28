import logging

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Models3d(nn.Module):
    def __init__(self, number_of_classes, window_size=1280, depthwise_multiplier=32, W=3, H=8):
        super(Models3d, self).__init__()
        self.logger = logging.getLogger()
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
        self._conv1 = nn.Conv3d(1, depthwise_multiplier, groups=1, kernel_size=(w_ker_siz, 3, 3))
        self._pool1 = nn.AvgPool3d(kernel_size=(1, 3, 1))
        # self._batch_norm1 = nn.BatchNorm3d(depthwise_multiplier)
        # self._prelu1 = nn.ReLU(depthwise_multiplier)
        # self._dropout1 = nn.Dropout3d(.5)

        # 1st Downsample time dimension
        if self._window:
            self._conv2 = nn.Conv2d(depthwise_multiplier, 2 * depthwise_multiplier, kernel_size=(w_ker_siz, 3),
                                    stride=(2, 2))
            self._pool2 = nn.AvgPool2d(kernel_size=(3, 3))
            # self._batch_norm2 = nn.BatchNorm2d(2 * depthwise_multiplier)
            # self._prelu2 = nn.ReLU(2 * depthwise_multiplier)
            # self._dropout2 = nn.Dropout2d(.5)

             # = nn.Conv2d(2*depthwise_multiplier, 4 * depthwise_multiplier, kernel_size=(w_ker_siz, 1),
             #                        stride=(2, 1))
            self._conv3 = nn.Conv1d(2*depthwise_multiplier, 4*depthwise_multiplier, kernel_size=w_ker_siz, stride=2)
            self._pool3 = nn.AvgPool1d(kernel_size=3)
            # self._batch_norm3 = nn.BatchNorm1d(4 * depthwise_multiplier)
            # self._prelu3 = nn.ReLU(4 * depthwise_multiplier)
            # self._dropout3 = nn.Dropout(.5)

            self._conv4 = nn.Conv1d(4 * depthwise_multiplier, 8 * depthwise_multiplier, kernel_size=w_ker_siz, stride=2)
            self._pool4 = nn.AvgPool1d(kernel_size=3)
            # self._batch_norm4 = nn.BatchNorm1d(8 * depthwise_multiplier)
            # self._prelu4 = nn.ReLU(8 * depthwise_multiplier)
            # self._dropout4 = nn.Dropout(.5)

        self.flatten = lambda x: x.view(-1, 8*depthwise_multiplier)
        # self._fc1 = nn.Linear(W*H, 8*depthwise_multiplier)
        self._fc1 = nn.Linear(8*depthwise_multiplier, 8*depthwise_multiplier)
        # self._batch_norm5 = nn.BatchNorm1d(8*depthwise_multiplier)
        # self._prelu5 = nn.ReLU(8*depthwise_multiplier)
        # self._dropout5 = nn.Dropout(.5)

        self._fc2 = nn.Linear(8 * depthwise_multiplier, 4 * depthwise_multiplier)
        # self._batch_norm6 = nn.BatchNorm1d(4 * depthwise_multiplier)
        # self._prelu6 = nn.ReLU(4 * depthwise_multiplier)
        # self._dropout6 = nn.Dropout(.5)

        self._fc3 = nn.Linear(4 * depthwise_multiplier, 2*depthwise_multiplier)
        # self._batch_norm6 = nn.BatchNorm1d(2 * depthwise_multiplier)
        # self._prelu7 = nn.ReLU(2*depthwise_multiplier)
        # self._dropout7 = nn.Dropout(.5)

        self._output = nn.Linear(2*depthwise_multiplier, number_of_classes)

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
        conv1 = F.relu(self._conv1(x))
        # conv1 = self._prelu1(self._conv1(x))
        # conv1 = self._dropout1(self._prelu1(self._conv1(x)))
        # conv1 = self._dropout1(self._prelu1(self._batch_norm1(self._conv1(x))))
        self.logger.debug(f'conv1 {conv1.shape}')
        pool1 = self._pool1(conv1)
        self.logger.debug(f'pool1 {pool1.shape}')


        if self._window:
            conv2 = F.relu(self._conv2(pool1.squeeze(dim=3)))
            # conv2 = self._dropout2(self._prelu2(self._batch_norm2(self._conv2(pool1.squeeze()))))
            self.logger.debug(f'conv2 {conv2.shape}')
            pool2 = self._pool2(conv2)
            self.logger.debug(f'pool2 {pool2.shape}')

            conv3 = self._conv3(pool2.squeeze(dim=3))
            self.logger.debug(f'conv3 {conv3.shape}')
            # conv3 = self._batch_norm3(conv3)
            # self.logger.debug(f'batchnorm {conv3.shape}')
            conv3 = F.relu(conv3)
            # conv3 = self._dropout3(self._prelu3(conv3))
            self.logger.debug(f'relu3 {conv3.shape}')
            # self.logger.debug(f'dropout3 {conv3.shape}')
            pool3 = self._pool3(conv3)
            self.logger.debug(f'pool3 {pool3.shape}')

            conv4 = self._conv4(pool3)
            self.logger.debug(f'conv4 {conv4.shape}')
            # conv4 = self._batch_norm4(conv4)
            # self.logger.debug(f'batchnorm {conv4.shape}')
            conv4 = F.relu(conv4)
            # conv4 = self._dropout4(self._prelu4(conv4))
            self.logger.debug(f'relu4 {conv4.shape}')
            # self.logger.debug(f'dropout4 {conv4.shape}')
            pool4 = self._pool3(conv4)
            self.logger.debug(f'pool4 {pool4.shape}')
        else:
            self.logger.debug('no time window')
            pool4 = pool1

        flatten_tensor = self.flatten(pool4)
        self.logger.debug(f'flatten_tensor {flatten_tensor.shape}')
        # if flatten_tensor.size(0) != x.size(0):
        #     print()
        fc1 = F.relu(self._fc1(flatten_tensor))
        # fc1 = self._prelu5(self._fc1(x))
        # fc1 = self._prelu5(self._fc1(flatten_tensor))
        # fc1 = self._dropout5(self._prelu5(self._fc1(flatten_tensor)))
        # fc1 = self._dropout5(self._prelu5(self._batch_norm5(self._fc1(flatten_tensor))))
        self.logger.debug(f'fc1 {fc1.shape}')
        fc2 = F.relu(self._fc2(fc1))
        # fc2 = self._prelu6(self._fc2(fc1))
        # fc2 = self._dropout6(self._prelu6(self._fc2(fc1)))
        # fc2 = self._dropout6(self._prelu6(self._batch_norm6(self._fc2(fc1))))
        self.logger.debug(f'fc2 {fc2.shape}')
        fc3 = F.relu(self._fc3(fc2))
        # fc3 = self._prelu7(self._fc3(fc2))
        # fc3 = self._dropout6(self._prelu7(self._fc3(fc2)))
        # fc3 = self._dropout7(self._prelu7(self._batch_norm7(self._fc3(fc2))))
        self.logger.debug(f'fc3 {fc3.shape}')
        output = self._output(fc3)
        self.logger.debug(f'logits {output.shape}')
        # output = F.softmax(output, dim=1)
        # self.logger.debug(f'softmax {output.shape}')
        return output

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(message)s')
    window = 259
    model = Models3d(number_of_classes=64, window_size=window, depthwise_multiplier=32)

    batch = 200

    channels = 1
    W = 3
    H = 8

    inp = torch.arange(batch * window * channels * W * H).float().reshape(batch, window, channels, W * H)

    l = model(inp)