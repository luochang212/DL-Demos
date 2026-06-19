# DDPM 章节

本目录是 DDPM 教程章节的源文件目录。网页教程、可运行代码、公式推导和测试都应与这里保持一致。

## 目录结构

- `code/`：训练、采样、数据加载、DDPM 公式和噪声预测网络实现。
- `derivations/`：完整公式推导和 Lean4 公式校验代码。
- `tests/`：章节级 CPU smoke tests。

## 运行命令

不依赖 MNIST 的真实 CLI 自检：

```powershell
uv run python -m chapters.generative_models.ddpm.code.main --mode smoke --device cpu
```

该命令会使用合成 28x28 灰度图片跑一次真实训练路径，保存 checkpoint，并生成一张样例图。
`--device auto` 会优先使用 CUDA，其次使用 macOS MPS，最后回退到 CPU；没有 GPU 也可以用 CPU 跑通自检。

## 数据准备

DDPM 训练使用 MNIST 手写数字图片。

- 数据来源：`torchvision.datasets.MNIST`
- 官方来源：[The MNIST Database](http://yann.lecun.com/exdb/mnist/)
- 默认路径：`data/mnist/`
- 数据规模：60,000 张训练图片，10,000 张测试图片，灰度 28x28。
- 使用限制：MNIST 是经典公开教学数据集，适合快速训练和观察扩散模型效果。

本章代码会通过 torchvision 读取 MNIST，并执行：

```python
ToTensor()
Lambda(lambda x: (x - 0.5) * 2)
```

也就是把像素从 `[0, 1]` 归一化到 `[-1, 1]`，与扩散采样输出范围保持一致。

快速训练和采样：

```powershell
uv run python -m chapters.generative_models.ddpm.code.main --mode train --epochs 1 --batch-size 128 --config-id 0 --n-steps 100 --device auto --checkpoint work_dirs/ddpm/mnist_model.pth
uv run python -m chapters.generative_models.ddpm.code.main --mode sample --config-id 0 --n-steps 100 --device auto --checkpoint work_dirs/ddpm/mnist_model.pth --output work_dirs/ddpm/mnist_sample.jpg --n-sample 16
```

正式配置可以使用默认 `unet_res_cfg` 和 `n_steps=1000`，但训练时间会明显更长。

## Checkpoint 和预训练模型

本章当前不依赖第三方预训练模型。训练和采样使用的是本项目代码训练出的 checkpoint。

- 默认 checkpoint：`work_dirs/ddpm/model_unet_res.pth`
- 自检 checkpoint：`work_dirs/ddpm/smoke_model.pth`
- 默认输出图片：`work_dirs/ddpm/diffusion.jpg`
- 自检输出图片：`work_dirs/ddpm/smoke_sample.jpg`

`--mode sample` 需要先有 checkpoint；如果还没有训练过，请先运行 `--mode smoke` 验证链路，或运行 `--mode train` 在 MNIST 上训练。

Hugging Face 上有大量扩散模型和预训练 VAE/UNet，但它们不是本章 DDPM 代码的必要依赖。磁盘空间不足时不要下载大模型；后续如果章节接入第三方预训练模型，需要在这里明确模型来源、大小、下载方式和本地路径。

## 验证命令

```powershell
uv run pytest chapters/generative_models/ddpm/tests -q
uv run ruff check chapters/generative_models/ddpm
lake build
```
