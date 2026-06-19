import numpy as np
import torch

from chapters.fundamentals.multiclass_classification.code.model import (
    MulticlassClassificationNet,
    train,
)
from chapters.fundamentals.multiclass_classification.code.points_classification import (
    generate_plot_set,
    generate_points,
    plot_points,
    visualize,
)


def main():
    train_X, train_Y = generate_points(400)
    plot_points(train_X, train_Y)
    plot_X = generate_plot_set()

    # X: [2, m]
    # Y: [1, m]

    train_X_pt = torch.tensor(train_X, dtype=torch.float32)
    train_Y_pt = torch.tensor(train_Y.squeeze(0), dtype=torch.long)

    print(train_X_pt.shape)
    print(train_Y_pt.shape)

    # X: [2, m]
    # Y: [m]

    n_x = 2
    neuron_list = [n_x, 10, 10, 3]
    model = MulticlassClassificationNet(neuron_list)
    train(model, train_X_pt, train_Y_pt, 5000, 0.001, 1000)

    plot_result = model.forward(torch.Tensor(plot_X))
    plot_result = torch.argmax(plot_result, 0).numpy()
    plot_result = np.expand_dims(plot_result, 0)

    visualize(train_X, train_Y, plot_result)


if __name__ == '__main__':
    main()
