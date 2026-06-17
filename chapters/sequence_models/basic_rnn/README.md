# Basic RNN

1. Download [IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

2. Modify the directory in `read_imdb`.

3. Train, evaluate, or sample explicitly:

```bash
uv run python chapters/sequence_models/basic_rnn/main.py --model rnn1 --mode train
uv run python chapters/sequence_models/basic_rnn/main.py --model rnn1 --mode evaluate
uv run python chapters/sequence_models/basic_rnn/main.py --model rnn1 --mode sample
```

The default device is selected automatically. Use `--device cpu` or
`--device cuda:0` to choose one explicitly, and `--checkpoint PATH` to override
the default `rnn1.pth` or `rnn2.pth` path. Training and evaluation require the
IMDb vocabulary data; sampling only requires a checkpoint.
