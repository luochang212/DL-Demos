# DL-Demos

从公式到代码的深度学习教学实现，配套中文 Docusaurus 教程站。

## 环境

默认环境为 Python 3.13 + PyTorch，使用 uv 管理：

```bash
uv sync --group dev
```

维护中的非分布式 PyTorch 示例会自动选择 CUDA 或 CPU。完整训练通常需要对应数据集，单元测试使用合成输入，不要求 GPU 或真实数据集。DDP 示例仍明确要求 CUDA/NCCL。

运行示例：

```bash
uv run python dldemos/ShallowNetwork/points_classification.py
uv run python dldemos/Transformer/train.py
uv run python dldemos/StyleTransfer/style_transfer.py --steps 50
```

质量检查：

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pre-commit run --all-files
```

## 教程内容

- 基础神经网络：Logistic Regression、浅层网络、深层 MLP、多分类
- 训练技巧：初始化、正则化、优化器
- CNN 与视觉：卷积、ResNet、Style Transfer
- 序列模型：RNN、情感分析、Attention、Transformer
- 生成模型：VAE、DDPM、DDIM、PixelCNN、VQVAE
- 工程实践：Fourier Feature、PyTorch DDP

教程站源码位于 `website/`：

```bash
cd website
npm ci
npm run build
```

## Legacy 示例

默认环境不包含 TensorFlow 和已停止维护的 `torchtext`。相关实现作为教学对照保留，但不属于默认 Python 3.13 环境的可运行性承诺。
