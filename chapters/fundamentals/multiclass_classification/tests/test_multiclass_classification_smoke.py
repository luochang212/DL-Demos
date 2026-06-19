from pathlib import Path

import torch

from chapters.fundamentals.multiclass_classification.code.model import (
    MulticlassClassificationNet,
)
from chapters.fundamentals.multiclass_classification.code.points_classification import (
    generate_plot_set,
    generate_points,
    visualize,
)


def test_multiclass_classifier_forward_loss_and_output_path():
    x, y = generate_points(12)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y.squeeze(0), dtype=torch.long)
    model = MulticlassClassificationNet([2, 4, 3])

    logits = model.forward(x_tensor)
    loss = model.loss(y_tensor, logits)
    loss.backward()

    output_path = Path('work_dirs/multiclass_classification/test_visualize.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_logits = model.forward(torch.tensor(generate_plot_set(), dtype=torch.float32))
    plot_result = torch.argmax(plot_logits, 0).numpy()
    visualize(x, y, plot_result, output_path=output_path)

    assert logits.shape == (3, 12)
    assert torch.isfinite(loss)
    assert output_path.exists()
