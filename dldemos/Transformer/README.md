# Transformer

This directory contains an educational English-to-Chinese Transformer.

```shell
uv run python dldemos/Transformer/data_load.py
uv run python dldemos/Transformer/train.py
uv run python dldemos/Transformer/translate.py
```

`load_train_data()` returns `(english_source, chinese_target)`. Training and translation automatically select CUDA or CPU.
