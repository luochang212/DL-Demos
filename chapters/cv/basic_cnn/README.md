1. Sync the project environment:

```shell
uv sync
```

2. Download the dataset from https://www.kaggle.com/datasets/fusicfenta/cat-and-dog?resource=download and organize the directory as follows:

```plain text
鈹斺攢data
    鈹斺攢archive
        鈹斺攢dataset
            鈹溾攢single_prediction
            鈹溾攢test_set
            鈹? 鈹溾攢cats
            鈹? 鈹斺攢dogs
            鈹斺攢training_set
                鈹溾攢cats
                鈹斺攢dogs
```

3. Modify the path in "main" scripts:

```Python
train_X, train_Y, test_X, test_Y = get_cat_set(
        'data/archive/dataset', train_size=1500)
```

Replace 'data/archive/dataset' with your path.

4. Run `uv run python chapters/cv/basic_cnn/pt_main.py`.

The TensorFlow implementation is retained as a legacy comparison.

The NumPy implementation of convolution is in `np_conv` and `np_conv_backward`
