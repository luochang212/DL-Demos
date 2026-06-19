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

## 参考资料

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- 周弈帆（2022-09-21）。[吴恩达《深度学习专项》笔记（十七）：Transformer](https://zhouyifan.net/2022/09/21/DLS-note-17/)。
- 周弈帆（2023-06-10）。[PyTorch Transformer 英中翻译超详细教程](https://zhouyifan.net/2023/06/10/20221106-transformer-pytorch/)。
- 周弈帆（2025-08-24）。[FlashAttention 教程（算法原理篇）](https://zhouyifan.net/2025/08/24/20250511-flashattention-1/)。
- 周弈帆（2024-12-08）。[位置编码长度外推技术](https://zhouyifan.net/2024/12/08/20241208-Context-Window-Extension/)。
- 周弈帆（2025-12-18）。[Log-linear Sparse Attention](https://zhouyifan.net/2025/12/18/20251211-llsa-1/)。
- 周弈帆（2024-12-04）。[位置编码背后的理论解释——傅里叶特征与核回归](https://zhouyifan.net/2024/12/04/20241202-fourier-feature/)。
- 苏剑林（2021）。[Transformer 升级之路：旋转位置编码](https://kexue.fm/archives/8265)。
- 完整公式推导：[derivations/formulas.md](derivations/formulas.md)

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
