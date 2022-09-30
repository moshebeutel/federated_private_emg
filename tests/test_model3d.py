import logging
# import unittest

import torch
from federated_private_emg.fed_priv_models.model3d import Model3d


# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         self.assertEqual(True, False)  # add assertion here


def test_model3d():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(message)s')
    window = 259
    model = Model3d(number_of_classes=64, window_size=window, depthwise_multiplier=32)

    batch = 200

    channels = 1
    W = 3
    H = 8

    inp = torch.arange(batch * window * channels * W * H).float().reshape(batch, window, channels, W * H)

    l = model(inp)

    assert l.shape == (200, 64)


# if __name__ == '__main__':
#     unittest.main()
