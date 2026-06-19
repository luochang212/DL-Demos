import numpy as np

from chapters.fundamentals.logistic_regression.code.model import (
    init_weights,
    loss,
    predict,
    train_step,
)


def test_logistic_regression_train_step_is_finite():
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([[0.0, 1.0]])
    w, b = init_weights(n_x=2)

    w, b = train_step(w, b, x, y, lr=0.1)
    y_hat = predict(w, b, x)

    assert np.isfinite(loss(y_hat, y))
    assert w.shape == (2, 1)
    assert isinstance(b, float) or np.isscalar(b)
