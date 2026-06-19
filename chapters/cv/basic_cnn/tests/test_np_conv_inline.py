import numpy as np
import pytest
import torch

from chapters.cv.basic_cnn.code.np_conv import conv2d
from chapters.cv.basic_cnn.code.np_conv_backward import (
    conv2d_backward,
    conv2d_forward,
)


@pytest.mark.parametrize('c_i, c_o', [(3, 6), (2, 2)])
@pytest.mark.parametrize('kernel_size', [3, 5])
@pytest.mark.parametrize('stride', [1, 2])
@pytest.mark.parametrize('padding', [0, 1])
@pytest.mark.parametrize('dilation', [1, 2])
@pytest.mark.parametrize('groups', ['1', 'all'])
@pytest.mark.parametrize('bias', [False, True])
def test_conv_forward(
    c_i: int,
    c_o: int,
    kernel_size: int,
    stride: int,
    padding: str,
    dilation: int,
    groups: str,
    bias: bool,
):
    if groups == '1':
        groups = 1
    elif groups == 'all':
        groups = c_i

    if bias:
        bias = np.random.randn(c_o)
        torch_bias = torch.from_numpy(bias)
    else:
        bias = None
        torch_bias = None

    input = np.random.randn(20, 20, c_i)
    weight = np.random.randn(c_o, kernel_size, kernel_size, c_i // groups)

    torch_input = torch.from_numpy(np.transpose(input, (2, 0, 1))).unsqueeze(0)
    torch_weight = torch.from_numpy(np.transpose(weight, (0, 3, 1, 2)))
    torch_output = torch.conv2d(
        torch_input, torch_weight, torch_bias, stride, padding, dilation, groups
    ).numpy()
    torch_output = np.transpose(torch_output.squeeze(0), (1, 2, 0))

    numpy_output = conv2d(input, weight, stride, padding, dilation, groups, bias)

    assert np.allclose(torch_output, numpy_output)


@pytest.mark.parametrize('c_i, c_o', [(3, 6), (2, 2)])
@pytest.mark.parametrize('kernel_size', [3, 5])
@pytest.mark.parametrize('stride', [1, 2])
@pytest.mark.parametrize('padding', [0, 1])
def test_conv_backward(c_i: int, c_o: int, kernel_size: int, stride: int, padding: str):

    # Preprocess
    input = np.random.randn(20, 20, c_i)
    weight = np.random.randn(c_o, kernel_size, kernel_size, c_i)
    bias = np.random.randn(c_o)

    torch_input = (
        torch.from_numpy(np.transpose(input, (2, 0, 1))).unsqueeze(0).requires_grad_()
    )
    torch_weight = torch.from_numpy(np.transpose(weight, (0, 3, 1, 2))).requires_grad_()
    torch_bias = torch.from_numpy(bias).requires_grad_()

    # forward
    torch_output_tensor = torch.conv2d(
        torch_input, torch_weight, torch_bias, stride, padding
    )
    torch_output = np.transpose(
        torch_output_tensor.detach().numpy().squeeze(0), (1, 2, 0)
    )

    cache = conv2d_forward(input, weight, bias, stride, padding)
    numpy_output = cache['Z']

    assert np.allclose(torch_output, numpy_output)

    # backward
    torch_sum = torch.sum(torch_output_tensor)
    torch_sum.backward()
    torch_dW = np.transpose(torch_weight.grad.numpy(), (0, 2, 3, 1))
    torch_db = torch_bias.grad.numpy()
    torch_dA_prev = np.transpose(torch_input.grad.numpy().squeeze(0), (1, 2, 0))

    dZ = np.ones(numpy_output.shape)
    dW, db, dA_prev = conv2d_backward(dZ, cache, stride, padding)

    assert np.allclose(dW, torch_dW)
    assert np.allclose(db, torch_db)
    assert np.allclose(dA_prev, torch_dA_prev)
