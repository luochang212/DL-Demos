from chapters.training_tricks.initialization.code.model import DeepNetwork, train
from chapters.training_tricks.initialization.code.points_classification import (
    generate_plot_set,
    generate_points,
    plot_points,
    visualize,
)


def main():
    train_X, train_Y = generate_points(400)
    plot_points(train_X, train_Y)
    plot_X = generate_plot_set()

    n_x = train_X.shape[0]
    neuron_list = [n_x, 10, 5, 1]
    activation_list = ['relu', 'relu']
    model1 = DeepNetwork(neuron_list, activation_list, 'zeros')
    model2 = DeepNetwork(neuron_list, activation_list, 'random')
    model3 = DeepNetwork(neuron_list, activation_list, 'he')
    train(model1, train_X, train_Y, 20000, 0.01, 1000)
    train(model2, train_X, train_Y, 20000, 0.01, 1000)
    train(model3, train_X, train_Y, 20000, 0.01, 1000)

    plot_result1 = model1.forward(plot_X, False)
    plot_result2 = model2.forward(plot_X, False)
    plot_result3 = model3.forward(plot_X, False)

    visualize(train_X, train_Y, plot_result1)
    visualize(train_X, train_Y, plot_result2)
    visualize(train_X, train_Y, plot_result3)


if __name__ == '__main__':
    main()
