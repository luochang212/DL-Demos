from pathlib import Path

import numpy as np

from chapters.fundamentals.deep_network.code.model import DeepNetwork, train


def test_deep_network_train_and_checkpoint_path():
    model = DeepNetwork([2, 3, 1], ['relu', 'sigmoid'])
    x = np.random.rand(2, 8)
    y = (x[0:1] > x[1:2]).astype(float)

    train(model, x, y, step=2, learning_rate=0.01, print_interval=10)
    output_path = Path('work_dirs/deep_network/test_model.npz')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    assert output_path.exists()
    assert np.isfinite(model.loss(y, model.forward(x, train_mode=False)))
