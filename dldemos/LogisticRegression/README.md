1. Download the dataset from https://www.kaggle.com/datasets/fusicfenta/cat-and-dog?resource=download and organize the directory as follows:

```plain text
└─data
    └─archive
        └─dataset
            ├─single_prediction
            ├─test_set
            │  ├─cats
            │  └─dogs
            └─training_set
                ├─cats
                └─dogs
```

2. Run from the repository root:

```shell
uv run python dldemos/LogisticRegression/main.py
```
