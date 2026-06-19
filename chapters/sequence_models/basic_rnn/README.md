# Basic RNN

字符级语言模型（character-level language model），展示 RNN 最核心的循环计算机制。
实现两种模型：手写 RNN（RNN1）和 PyTorch `nn.GRU`（RNN2），均在 IMDB 词汇表上训练。

## 目录结构

```text
basic_rnn/
  README.md
  __init__.py
  code/
    __init__.py          # 导出 RNN1, RNN2
    constant.py          # 字符常量（27 个字符的映射表）
    model.py             # RNN1（手动循环 + one-hot）和 RNN2（nn.GRU + Embedding）
    dataset.py           # WordDataset, DataLoader 工厂, one-hot/标签转换
    main.py              # CLI 入口（train / evaluate / sample / smoke）
  derivations/
    formulas.md           # RNN 前向公式、BPTT、梯度裁剪、GRU 门控方程的完整推导
    basic_rnn.lean        # Lean 4 恒等式检查
  read_imdb.py            # IMDB 数据集读取工具
  tests/
    test_basic_rnn_smoke.py  # CPU 冒烟测试
```

## 运行命令

### Smoke（无额外数据依赖，CPU 快速自检）

```shell
uv run python -m chapters.sequence_models.basic_rnn.code.main --mode smoke
```

Smoke 模式使用 IMDB 词汇表数据训练 1 个 batch，评估语言模型并采样 20 个单词。

### 训练

```shell
# 训练 RNN1（手动循环，one-hot 输入）
uv run python -m chapters.sequence_models.basic_rnn.code.main --model rnn1 --mode train

# 训练 RNN2（nn.GRU + Embedding）
uv run python -m chapters.sequence_models.basic_rnn.code.main --model rnn2 --mode train

# 指定设备
uv run python -m chapters.sequence_models.basic_rnn.code.main --model rnn1 --mode train --device cuda:0
```

### 评估

```shell
uv run python -m chapters.sequence_models.basic_rnn.code.main --model rnn1 --mode evaluate
```

评估脚本对一组手工设计的"单词对"（如 apple/appll、bear/beer）计算语言模型概率，
验证正确拼写的概率是否高于错误拼写。

### 采样

```shell
uv run python -m chapters.sequence_models.basic_rnn.code.main --model rnn2 --mode sample
```

从语言模型概率分布中随机采样 20 个"单词"，观察模型是否学到了英文的拼写规律。

### 查看帮助

```shell
uv run python -m chapters.sequence_models.basic_rnn.code.main --help
```

## 数据准备

本教程使用 [IMDb 数据集](https://ai.stanford.edu/~amaas/data/sentiment/)，由 Stanford
发布，用于非商业研究。

- **下载地址**：https://ai.stanford.edu/~amaas/data/sentiment/
- **本地路径**：`data/aclImdb/`（需手动下载并解压）
- **所需文件**：`data/aclImdb/imdb.vocab`（词汇表文件）即可满足字符级语言模型训练
- **大小**：词汇表约 90K 单词，磁盘占用约 1MB
- **格式**：每行一个单词的纯文本文件

训练时 `get_dataloader_and_max_length()` 通过 `read_imdb_vocab()` 读取词汇表，
不直接使用影评文件。

## Checkpoint 和预训练模型

- 训练后 checkpoint 默认输出到 `work_dirs/basic_rnn/rnn1.pth` 或 `rnn2.pth`
- 可通过 `--checkpoint PATH` 覆盖默认路径
- 评估和采样模式需要已有 checkpoint，否则会报 `FileNotFoundError`
- RNN1 checkpoint 约 50KB，RNN2 checkpoint 约 200KB

## 验证命令

### 冒烟测试

```shell
uv run pytest chapters/sequence_models/basic_rnn/tests -q
```

### 代码质量

```shell
uv run ruff check chapters/sequence_models/basic_rnn
uv run ruff format --check chapters/sequence_models/basic_rnn
```

### 公式验证

```shell
lake build
```

### 网站构建

```shell
cd website && npm run build
```
