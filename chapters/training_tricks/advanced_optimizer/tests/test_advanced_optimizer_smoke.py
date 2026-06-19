from pathlib import Path

import numpy as np

from chapters.training_tricks.advanced_optimizer.code.model import DeepNetwork, train
from chapters.training_tricks.advanced_optimizer.code.optimizer import Adam


def test_adam_train_step_writes_checkpoint(tmp_path):
    model = DeepNetwork([2, 3, 1], ['relu', 'sigmoid'])
    optimizer = Adam(model.save(), learning_rate=0.01)
    x = np.random.rand(2, 8)
    y = (x[0:1] > x[1:2]).astype(float)
    save_dir = tmp_path / 'optimizer'

    train(
        model,
        optimizer,
        x,
        y,
        total_epoch=2,
        batch_size=4,
        save_dir=str(save_dir),
        print_interval=10,
    )

    assert Path(save_dir, 'model_latest.npz').exists()
    assert np.isfinite(model.loss(y, model.forward(x, train_mode=False)))
