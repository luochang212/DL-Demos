# Transformer

完整的 Encoder-Decoder Transformer 实现（Vaswani et al., 2017），用于英语到中文的
机器翻译。包含 Multi-Head Attention、位置编码、三种 Mask 策略和自回归贪心解码。

## 目录结构

```text
transformer/
  README.md
  __init__.py
  code/
    __init__.py          # 导出 Transformer
    model.py             # PositionalEncoding, MultiHeadAttention, Encoder, Decoder, Transformer
    data_load.py         # 语料下载、词表构建、数据加载
    main.py              # CLI 入口（train / translate / smoke）
  derivations/
    formulas.md           # Scaled Dot-Product, Multi-Head, PE, FFN, LayerNorm 的完整推导
    transformer.lean      # Lean 4 恒等式检查
  model.py, data_load.py, train.py, translate.py  # 旧版文件（向后兼容）
  tests/
    test_transformer_smoke.py  # CPU 冒烟测试（4 个测试）
```

## 运行命令

### Smoke（下载数据 + 训练 1 epoch + 翻译，CPU 可运行）

```shell
uv run python -m chapters.sequence_models.transformer.code.main --mode smoke
```

首次运行会从 GitHub 下载平行语料和预构建词表。

### 下载数据

```shell
uv run python -m chapters.sequence_models.transformer.code.data_load
```

### 训练

```shell
uv run python -m chapters.sequence_models.transformer.code.main --mode train --epochs 60
```

默认配置：6 层 Encoder + 6 层 Decoder，8 头注意力，$d_{\text{model}} = 512$，$d_{ff} = 2048$。

### 翻译

```shell
uv run python -m chapters.sequence_models.transformer.code.main --mode translate
```

默认翻译 "we should protect environment" → 中文。

### 查看帮助

```shell
uv run python -m chapters.sequence_models.transformer.code.main --help
```

## 数据说明

- **语料来源**：从 GitHub 自动下载（`P3n9W31/transformer-pytorch` 仓库的预构建数据）
- **训练数据**：英文-中文平行语料，存储于 `chapters/sequence_models/transformer/data/`
- **词表**：预构建的中英文词表（`.vocab.tsv`），包含 `<S>`, `</S>`, `<UNK>`, `<PAD>` 特殊标记
- **序列长度**：默认截断至 `maxlen = 50`
- **数据契约**：`load_train_data()` 返回 `(english_source, chinese_target)`

## Checkpoint 和预训练模型

- 训练后 checkpoint 默认输出到 `work_dirs/transformer/model.pth`
- 可通过 `--checkpoint PATH` 覆盖
- 模型约 40MB（6 层，512 维）
- 训练 60 epoch 在训练集上可达较好效果

## 验证命令

### 冒烟测试

```shell
uv run pytest chapters/sequence_models/transformer/tests -q
```

### 代码质量

```shell
uv run ruff check chapters/sequence_models/transformer
uv run ruff format --check chapters/sequence_models/transformer
```

### 公式验证

```shell
lake build
```

### 网站构建

```shell
cd website && npm run build
```
