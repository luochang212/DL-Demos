# DDIM：去噪扩散隐式模型

本章是在 DDPM 之后的采样加速章节。DDIM 不改变训练目标，仍然训练网络预测噪声
`eps_theta(x_t, t)`；它改变的是反向采样过程，让模型可以从较少的时间步直接跳采样。

## 目录结构

```text
chapters/generative_models/ddim/
  README.md
  code/
  derivations/
  tests/
```

- `code/`：DDIM 的模型、DDPM 基类、数据加载和命令行入口。
- `derivations/formulas.md`：人可读的完整公式推导。
- `derivations/ddim.lean`：用 Lean4 检查公式中的基础代数关系。
- `tests/`：CPU smoke test，不依赖真实数据集。

## 快速运行

先运行不下载数据的 smoke，确认模型、损失、反传、checkpoint 和采样输出都能走通：

```powershell
uv run python -m chapters.generative_models.ddim.code.main --help
uv run python -m chapters.generative_models.ddim.code.main --mode smoke --device cpu
```

默认会写入：

```text
work_dirs/ddim/smoke_model.pth
work_dirs/ddim/smoke_sample.jpg
```

## 数据

MNIST 会下载到仓库外层约定目录：

```text
data/mnist/
```

训练命令：

```powershell
uv run python -m chapters.generative_models.ddim.code.main --mode train --device auto --data-root data/mnist --epochs 1 --batch-size 128
```

MNIST 来自 `torchvision.datasets.MNIST`。本仓库不提交 `data/` 下的数据文件。

## Checkpoint 与采样

训练默认 checkpoint：

```text
work_dirs/ddim/mnist.pth
```

用训练好的 DDPM/DDIM 噪声预测网络做 DDIM 采样：

```powershell
uv run python -m chapters.generative_models.ddim.code.main --mode sample --device auto --checkpoint work_dirs/ddim/mnist.pth --output work_dirs/ddim/sample.jpg --ddim-step 20 --eta 0
```

本章暂不内置预训练权重。若使用 Hugging Face 或其他来源的预训练模型，请在章节说明里补充来源、许可证、文件大小、放置路径和加载命令；权重文件不要提交到 git。

## 公式推导

阅读完整推导：

```text
chapters/generative_models/ddim/derivations/formulas.md
```

运行 Lean4 验证：

```powershell
lake build
```

## 验收命令

```powershell
uv run ruff check chapters/generative_models/ddim
uv run ruff format --check chapters/generative_models/ddim
uv run pytest chapters/generative_models/ddim/tests -q
uv run python -m chapters.generative_models.ddim.code.main --mode smoke --device cpu
lake build
```

## 参考资料

- Jiaming Song, Chenlin Meng, Stefano Ermon. Denoising Diffusion Implicit Models. ICLR 2021.
- Jonathan Ho, Ajay Jain, Pieter Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS 2020.
- 周一帆：[DDIM 是什么？](https://zhouyifan.net/2023/07/07/20230702-DDIM/)
