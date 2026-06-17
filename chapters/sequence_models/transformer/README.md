# Transformer

This directory contains an educational English-to-Chinese Transformer.

```shell
uv run python chapters/sequence_models/transformer/data_load.py
uv run python chapters/sequence_models/transformer/train.py
uv run python chapters/sequence_models/transformer/translate.py
```

`load_train_data()` returns `(english_source, chinese_target)`. Training and
translation automatically select CUDA or CPU. Translation adds the same source
boundary tokens used during training and maps unknown words to `<UNK>`.
