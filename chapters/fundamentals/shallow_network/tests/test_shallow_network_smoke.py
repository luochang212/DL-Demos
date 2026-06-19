from pathlib import Path

import numpy as np

from chapters.fundamentals.shallow_network.code.model import ShallowNetwork
from chapters.fundamentals.shallow_network.genereate_points import visualize


def test_shallow_network_forward_backward_and_output_path():
    model = ShallowNetwork(2, 4)
    x = np.random.rand(2, 8)
    y = (x[0:1] > x[1:2]).astype(float)

    y_hat = model.forward(x)
    model.backward(y)
    model.gradient_descent(0.01)

    output_path = Path('work_dirs/shallow_network/test_visualize.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualize(x.T, y.T, np.zeros((10000,)), output_path=output_path)

    assert y_hat.shape == y.shape
    assert output_path.exists()
