import numpy as np

from chapters.training_tricks.initialization.main import DeepNetwork, train


def test_initialization_network_train_step_is_finite():
    model = DeepNetwork([2, 3, 1], ['relu'], initialization='he')
    x = np.random.rand(2, 8)
    y = (x[0:1] > x[1:2]).astype(float)

    train(model, x, y, step=2, learning_rate=0.01, print_interval=10)

    assert np.isfinite(model.loss(y, model.forward(x, train_mode=False)))
