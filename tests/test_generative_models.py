import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from dldemos.Regularization.main import DeepNetwork
from dldemos.ddim.ddim import DDIM
from dldemos.pixelcnn.model import GatedPixelCNN, PixelCNN


class ZeroNoisePredictor(nn.Module):

    def forward(self, x, t):
        return torch.zeros_like(x)


class DDIMTest(unittest.TestCase):

    def test_eta_scales_standard_deviation_linearly(self):
        ddim = DDIM('cpu', 4)
        initial = torch.zeros(1, 1, 1, 1)
        net = ZeroNoisePredictor()

        with patch('torch.randn_like', side_effect=lambda x: torch.ones_like(x)):
            eta_one = ddim.sample_backward(initial,
                                           net,
                                           'cpu',
                                           simple_var=False,
                                           ddim_step=2,
                                           eta=1)
        with patch('torch.randn_like', side_effect=lambda x: torch.ones_like(x)):
            eta_half = ddim.sample_backward(initial,
                                            net,
                                            'cpu',
                                            simple_var=False,
                                            ddim_step=2,
                                            eta=0.5)

        torch.testing.assert_close(eta_half, eta_one * 0.5)


class PixelCNNTest(unittest.TestCase):

    def test_batch_norm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'leaks future pixels'):
            PixelCNN(1, 2, 2, bn=True, color_level=2)
        with self.assertRaisesRegex(ValueError, 'leaks future pixels'):
            GatedPixelCNN(1, 2, 2, bn=True, color_level=2)

    def test_future_pixels_do_not_affect_current_prediction(self):
        torch.manual_seed(0)
        model = GatedPixelCNN(1, 2, 2, color_level=2).train()
        x = torch.randn(1, 1, 5, 5, requires_grad=True)

        model(x)[0, 0, 2, 2].backward()

        future_grad = x.grad[0, 0, 2, 3:].abs().sum()
        future_grad += x.grad[0, 0, 3:, :].abs().sum()
        self.assertEqual(future_grad.item(), 0)


class RegularizationTest(unittest.TestCase):

    def test_dropout_mask_is_applied_during_backward(self):
        model = DeepNetwork([1, 2, 1], ['relu'], 'dropout')
        model.W[0] = np.ones((2, 1))
        model.W[1] = np.ones((1, 2))
        model.b[0] = np.zeros((2, 1))
        model.b[1] = np.zeros((1, 1))

        with patch('numpy.random.rand',
                   return_value=np.array([[0.0], [1.0]])):
            model.forward(np.ones((1, 1)))
        model.backward(np.zeros((1, 1)))

        self.assertEqual(model.dW_cache[0][1, 0], 0)


if __name__ == '__main__':
    unittest.main()
