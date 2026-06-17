import os

from chapters.fundamentals.deep_network.dataset import get_cat_set
from chapters.fundamentals.deep_network.model import DeepNetwork, train


def main():
    os.makedirs('work_dirs/deep_network', exist_ok=True)
    train_X, train_Y, test_X, test_Y = get_cat_set(
        'data/archive/dataset', train_size=1500
    )
    n_x = train_X.shape[0]
    model = DeepNetwork(
        [n_x, 30, 30, 20, 20, 1], ['relu', 'relu', 'relu', 'relu', 'sigmoid']
    )
    model_path = 'work_dirs/deep_network/model.npz'
    if os.path.exists(model_path):
        model.load(model_path)
    train(
        model,
        train_X,
        train_Y,
        500,
        learning_rate=0.01,
        print_interval=10,
        test_X=test_X,
        test_Y=test_Y,
    )
    model.save(model_path)


if __name__ == '__main__':
    main()
