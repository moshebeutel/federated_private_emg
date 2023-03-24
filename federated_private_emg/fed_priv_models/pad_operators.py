import torch
import torch.nn as nn
import torch.nn.functional as F


class NetWrapper:
    def __init__(self, net):
        self.net = net

    @property
    def net(self):
        return self._net

    @net.setter
    def net(self, value):
        self._net = value

class PadLastDimCircular(nn.Module):
    def __init__(self, window_size, W=3):
        super(PadLastDimCircular, self).__init__()
        # self._window_size = window_size
        # self._W = W
        self._pad_last_dim_circular = lambda x: \
            torch.concat([x[:, :, :, :, -1].reshape(-1, 1, window_size + 2, W + 2, 1),
                          x, x[:, :, :, :, 0].reshape(-1, 1, window_size + 2, W + 2, 1)], axis=4)

    def forward(self, x):
        # x1 = x[:, :, :, :, -1]
        # x2 = x[:, :, :, :, 0]
        # print(x1.shape, x.shape, x2.shape)
        # x1 = x1.reshape(-1, 1, self._window_size + 2, self._W + 2, 1)
        #
        #
        # x2 = x2.reshape(-1, 1, self._window_size + 2, self._W + 2, 1)

        # return torch.cat((x1, x, x2), dim=4)
        # print(x.shape)
        x = self._pad_last_dim_circular(x)
        return x
        # x = torch.concat([x[:, :, :, :, -1].reshape(-1, 1, self._window_size + 2 * self._hw_ker_siz, self._W + 2, 1),
        #                      x, x[:, :, :, :, 0].reshape(-1, 1, self._window_size + 2 * self._hw_ker_siz, self._W + 2, 1)], axis=4)


class PadBeforeLast(nn.Module):
    def __init__(self):
        super(PadBeforeLast, self).__init__()

    def forward(self, x):
        x = F.pad(x, pad=(0, 0, 1, 1, 1, 1), mode='constant')
        # print('PadBeforeLast', x.shape)
        return x


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()


class Reshape3Bands(nn.Module):

    def __init__(self, window_size, W=3, H=8):
        super(Reshape3Bands, self).__init__()
        self._window_size = window_size
        self._W = W
        self._H = H

    def forward(self, x):
        # print('Reshape3Bands', x.shape)
        x = x.reshape(x.shape[0], 1, self._window_size, self._W, self._H)
        # print('Reshape3Bands', x.shape)
        return x


class FlattenToLinear(nn.Module):
    def __init__(self, depthwise_multiplier=32):
        super(FlattenToLinear, self).__init__()
        self._depthwise_multiplier = depthwise_multiplier

    def forward(self, x):
        return x.view(-1, 8 * self._depthwise_multiplier)
