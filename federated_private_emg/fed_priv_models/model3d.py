import logging

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from fed_priv_models.nn_blocks import Conv3DBlock, Conv2DBlock, Conv1DBlock, DenseBlock


class Model3d(nn.Module):
    def __init__(self, number_of_classes,
                 window_size=1280,
                 depthwise_multiplier=32,
                 W=3,
                 H=8,
                 use_batch_norm=False,
                 use_dropout=False,
                 output_info_fn=print, output_debug_fn=None):
        super(Model3d, self).__init__()
        self._output_info_fn = lambda s: None if output_info_fn is None else output_info_fn
        self._output_debug_fn = lambda s: None if output_debug_fn is None else output_debug_fn
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
                                        use_batchnorm=use_batch_norm, use_dropout=use_dropout)

        # Downsample time dimension
        if self._window:
            self._conv_block2 = Conv2DBlock(depthwise_multiplier, 2 * depthwise_multiplier,
                                            kernel_size=(w_ker_siz, 3),
                                            stride=(2, 2),
                                            pool_kernel_size=(3, 3), use_batchnorm=use_batch_norm, use_dropout=use_dropout)

            self._conv_block3 = Conv1DBlock(2 * depthwise_multiplier, 4 * depthwise_multiplier,
                                            kernel_size=w_ker_siz,
                                            stride=2,
                                            pool_kernel_size=3, use_batchnorm=use_batch_norm, use_dropout=use_dropout)

            self._conv_block4 = Conv1DBlock(4 * depthwise_multiplier, 8 * depthwise_multiplier,
                                            kernel_size=w_ker_siz,
                                            stride=2,
                                            pool_kernel_size=3, use_batchnorm=use_batch_norm, use_dropout=use_dropout)

        self.flatten = lambda x: x.view(-1, 8 * depthwise_multiplier)

        self._dense_block1 = DenseBlock(8 * depthwise_multiplier, 8 * depthwise_multiplier,
                                        use_batchnorm=use_batch_norm, use_dropout=use_dropout)

        self._dense_block2 = DenseBlock(8 * depthwise_multiplier, 4 * depthwise_multiplier,
                                        use_batchnorm=use_batch_norm, use_dropout=use_dropout)

        self._dense_block3 = DenseBlock(4 * depthwise_multiplier, 2 * depthwise_multiplier,
                                        use_batchnorm=use_batch_norm, use_dropout=use_dropout)

        self._output = nn.Linear(2 * depthwise_multiplier, number_of_classes)

        self.initialize_weights()

        self._output_info_fn(str(self))

        self._output_info_fn("Number Parameters: {self.get_n_params()}")

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
        self._output_debug_fn(f'input {x.shape}')

        x = self._reshape(x)
        self._output_debug_fn(f'reshape {x.shape}')

        x = self._pad_before_last_dim_constant(x)
        self._output_debug_fn(f'_pad_before_last_dim_constant {x.shape}')

        x = self._pad_last_dim_circular(x)
        self._output_debug_fn(f'pad_last_dim {x.shape}')

        conv1 = self._conv_block1(x)
        self._output_debug_fn(f'conv1 {conv1.shape}')

        if self._window:

            conv2 = self._conv_block2(conv1.squeeze(dim=3))
            self._output_debug_fn(f'conv2 {conv2.shape}')

            conv3 = self._conv_block3(conv2.squeeze(dim=3))
            self._output_debug_fn(f'conv3 {conv3.shape}')

            conv4 = self._conv_block4(conv3)
            self._output_debug_fn(f'conv4 {conv4.shape}')

        else:
            self._output_debug_fn('no time window')
            conv4 = conv1
        flatten_tensor = self.flatten(conv4)
        self._output_debug_fn(f'flatten_tensor {flatten_tensor.shape}')

        fc1 = self._dense_block1(flatten_tensor)
        self._output_debug_fn(f'fc1 {fc1.shape}')

        fc2 = self._dense_block2(fc1)
        self._output_debug_fn(f'fc2 {fc2.shape}')

        fc3 = self._dense_block3(fc2)
        self._output_debug_fn(f'fc3 {fc3.shape}')

        output = self._output(fc3)
        self._output_debug_fn(f'logits {output.shape}')

        # output = F.softmax(output, dim=1)
        # self._output_debug_fn(f'softmax {output.shape}')

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
