# Attention

使用注意力机制的日期格式翻译模型（Bahdanau Attention），将各种格式的日期字符串
统一转换为 `yyyy-mm-dd` 标准格式。双向 LSTM 作为 Encoder，带注意力上下文的
单向 LSTM 作为 Decoder。

## 目录结构

```text
attention/
  README.md
  __init__.py
  code/
    __init__.py          # 导出 AttentionModel
    model.py             # AttentionModel（Bi-LSTM Encoder + Attention Decoder）
    dataset.py           # 日期数据生成、DateDataset、DataLoader 工厂
    main.py              # CLI 入口（train / test / infer / smoke）
  dataset.py              # 旧版日期生成工具（向后兼容）
  main.py                 # 旧版单文件实现（向后兼容）
  derivations/
    formulas.md           # 对齐分数、Softmax 归一化、上下文向量、交叉熵损失的完整推导
    attention.lean        # Lean 4 恒等式检查
  tests/
    test_attention_smoke.py  # CPU 冒烟测试
```

## 运行命令

### Smoke（自动生成小数据集，CPU 快速自检）

```shell
uv run python -m chapters.sequence_models.attention.code.main --mode smoke
```

Smoke 模式生成 5000 条训练数据和 1000 条测试数据，训练 2 个 epoch 后
输出测试准确率及推理示例。

### 训练

```shell
uv run python -m chapters.sequence_models.attention.code.main --mode train --epochs 30
```

首次运行会自动生成 `train.txt`（50000 条）和 `test.txt`（10000 条）。

### 测试

```shell
uv run python -m chapters.sequence_models.attention.code.main --mode test
```

### 推理

```shell
uv run python -m chapters.sequence_models.attention.code.main --mode infer
```

### 查看帮助

```shell
uv run python -m chapters.sequence_models.attention.code.main --help
```

## 数据说明

日期数据由 `Faker` 和 `babel` 库自动生成，无需额外下载。

- **训练集**：50000 条随机日期，`chapters/sequence_models/attention/train.txt`
- **测试集**：10000 条随机日期，`chapters/sequence_models/attention/test.txt`
- **格式**：每行 `输入日期\t标准日期`（TSV）
- **输入变体**：10 种格式（short/medium/long/full 及自定义格式）
- **标准输出**：固定 10 个字符的 `yyyy-mm-dd`
- **字符集**：ASCII 前 128 个字符（`EMBEDDING_LENGTH = 128`）

## Checkpoint 和预训练模型

- 训练后 checkpoint 默认输出到 `work_dirs/attention/model.pth`
- 可通过 `--checkpoint PATH` 覆盖
- 模型约 1MB
- 训练 30 epoch 在测试集可达 ~98% 准确率

## 验证命令

### 冒烟测试

```shell
uv run pytest chapters/sequence_models/attention/tests -q
```

### 代码质量

```shell
uv run ruff check chapters/sequence_models/attention
uv run ruff format --check chapters/sequence_models/attention
```

### 公式验证

```shell
lake build
```

### 网站构建

```shell
cd website && npm run build
```
