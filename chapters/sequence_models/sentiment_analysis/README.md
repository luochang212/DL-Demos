# Sentiment Analysis

基于 GloVe 词嵌入和 GRU 的 IMDB 影评情感二分类器。展示预训练词嵌入在 NLP 下游
任务中的迁移学习方法。

## 目录结构

```text
sentiment_analysis/
  README.md
  __init__.py
  code/
    __init__.py          # 导出 RNN
    model.py             # RNN 分类模型（GRU + GloVe → Sigmoid）
    dataset.py           # IMDBDataset, DataLoader 工厂（惰性加载 GloVe）
    main.py              # CLI 入口（train / test / infer / smoke）
  derivations/
    formulas.md           # GloVe 嵌入、GRU 门控、BCE 损失、变长批处理的完整推导
    sentiment_analysis.lean  # Lean 4 恒等式检查
  glove.py                # GloVe 类比推理演示（king-man+woman=queen）
  read_imdb.py            # IMDB 数据集读取工具
  main.py                 # 旧版单文件实现（向后兼容）
  tests/
    test_sentiment_analysis_smoke.py  # CPU 冒烟测试（mock torchtext）
```

## 运行命令

### Smoke（需要 IMDB 数据和 GloVe 下载，首次运行耗时较长）

```shell
uv run python -m chapters.sequence_models.sentiment_analysis.code.main --mode smoke
```

Smoke 模式训练 2 个 epoch，输出测试准确率和推理示例。
首次运行会下载 GloVe 词嵌入（~800MB）到 torchtext 缓存目录。

### 训练

```shell
uv run python -m chapters.sequence_models.sentiment_analysis.code.main --mode train --epochs 100
```

### 测试

```shell
uv run python -m chapters.sequence_models.sentiment_analysis.code.main --mode test
```

### 推理

```shell
uv run python -m chapters.sequence_models.sentiment_analysis.code.main --mode infer
```

### 查看帮助

```shell
uv run python -m chapters.sequence_models.sentiment_analysis.code.main --help
```

## 数据准备

本教程使用两个数据源：

### IMDB 数据集

- **下载地址**：https://ai.stanford.edu/~amaas/data/sentiment/
- **本地路径**：`data/aclImdb/`
- **大小**：约 80MB 压缩包
- **结构**：`train/pos/`, `train/neg/`, `test/pos/`, `test/neg/`, `imdb.vocab`

### GloVe 预训练词嵌入

- **来源**：[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- **版本**：6B tokens, 100 维
- **加载方式**：通过 `torchtext.vocab.GloVe` 自动下载
- **缓存位置**：`~/.cache/torch/text/` 或 torchtext 默认缓存

## 依赖说明

本章需要 `torchtext`（不在默认依赖中）：

```shell
uv add torchtext  # 或 pip install torchtext
```

冒烟测试通过 mock torchtext 可以在不安装该依赖的情况下运行。

## Checkpoint 和预训练模型

- 训练后 checkpoint 默认输出到 `work_dirs/sentiment_analysis/rnn.pth`
- 可通过 `--checkpoint PATH` 覆盖
- 模型约 500KB
- 训练 100 epoch 在测试集可达 ~90% 准确率

## 验证命令

### 冒烟测试

```shell
uv run pytest chapters/sequence_models/sentiment_analysis/tests -q
```

### 代码质量

```shell
uv run ruff check chapters/sequence_models/sentiment_analysis
uv run ruff format --check chapters/sequence_models/sentiment_analysis
```

### 公式验证

```shell
lake build
```

### 网站构建

```shell
cd website && npm run build
```
