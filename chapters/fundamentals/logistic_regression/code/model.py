"""Logistic Regression — core model functions.

Binary classification with sigmoid activation and binary cross-entropy loss.
All functions are pure NumPy, no framework dependencies beyond numpy.
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation: σ(x) = 1 / (1 + exp(-x))."""
    return 1 / (1 + np.exp(-x))


def init_weights(n_x: int = 224 * 224 * 3):
    """Initialise weights to zero and bias to zero.

    Returns:
        w: shape (n_x, 1)
        b: scalar float
    """
    w = np.zeros((n_x, 1))
    b = 0.0
    return w, b


def predict(w: np.ndarray, b: float, X: np.ndarray) -> np.ndarray:
    """Forward pass: ŷ = σ(wᵀX + b).

    Args:
        w: shape (d, 1)
        b: scalar
        X: shape (d, m) — samples as columns

    Returns:
        ŷ: shape (1, m) — predicted probabilities
    """
    return sigmoid(np.dot(w.T, X) + b)


def loss(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Binary cross-entropy loss.

    L = -mean(y·log(ŷ) + (1-y)·log(1-ŷ))

    Probabilities are clipped to [ε, 1-ε] to avoid log(0).
    """
    y_hat = np.clip(y_hat, 1e-12, 1 - 1e-12)
    return float(np.mean(-(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))))


def train_step(w: np.ndarray, b: float, X: np.ndarray, Y: np.ndarray, lr: float):
    """One step of gradient descent.

    The sigmoid + BCE combination simplifies the gradient w.r.t. logits to ŷ − y,
    avoiding the sigmoid saturation problem.

    Args:
        w: shape (d, 1)
        b: scalar
        X: shape (d, m)
        Y: shape (1, m)
        lr: learning rate

    Returns:
        Updated (w, b).
    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    dZ = A - Y
    dw = np.dot(X, dZ.T) / m
    db = np.mean(dZ)
    return w - lr * dw, b - lr * db
