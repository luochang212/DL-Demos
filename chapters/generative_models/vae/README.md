# VAE 章节

本目录是 VAE 教程章节的源文件目录。网页教程、可运行代码、公式推导、实验配置和测试都应与这里保持一致。

## 目录结构

- `code/`：网页教程对应的 PyTorch 可运行实现。
- `derivations/`：完整公式推导和 Lean4 公式校验代码。
- `tests/`：章节级 CPU smoke tests。

## 运行命令

不依赖 CelebA 的真实 CLI 自检：

```powershell
uv run python -m chapters.generative_models.vae.code.main --mode smoke --device cpu
```

该命令会使用合成图片跑一次真实训练路径，保存 checkpoint，并生成一张样例图。
`--device auto` 会优先使用 CUDA，其次使用 macOS MPS，最后回退到 CPU；没有 GPU 也可以用 CPU 跑通自检。

## 数据准备

VAE 正式训练使用 CelebA 的对齐人脸图片；如果只想快速看到模型学习效果，可以先用 MNIST。

### MNIST 快速效果

- 数据来源：`torchvision.datasets.MNIST`
- 默认路径：`data/mnist/`
- 用途：快速训练和观察数字重建/生成效果，不代表最终人脸生成质量。
- 处理方式：`Resize((64, 64))`、转成 3 通道、`ToTensor()`，以复用同一个 VAE 网络结构。

```powershell
uv run python -m chapters.generative_models.vae.code.main --mode train --dataset mnist --epochs 1 --device auto --checkpoint work_dirs/vae/mnist_model.pth
uv run python -m chapters.generative_models.vae.code.main --mode generate --device auto --checkpoint work_dirs/vae/mnist_model.pth --output work_dirs/vae/mnist_sample.jpg
```

### CelebA 正式数据

VAE 正式训练使用 CelebA 的对齐人脸图片。

- 官方来源：[CelebA Dataset - MMLab](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- 可用镜像：[Kaggle CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- 数据规模：官方说明包含 202,599 张人脸图片和 40 个属性标注。
- 使用限制：CelebA 主要用于非商业研究；下载和使用时应遵守原始数据集许可。

本章代码只读取图片，不使用属性标注。下载并解压 `img_align_celeba.zip` 后，把图片直接放到：

```text
data/celebA/img_align_celeba/
```

目录示例：

```text
data/
  celebA/
    img_align_celeba/
      000001.jpg
      000002.jpg
      000003.jpg
      ...
```

图片格式要求：

- 常见图片格式即可，推荐 `.jpg`。
- 每张图片会被 `PIL.Image.open(...).convert('RGB')` 读成 RGB。
- 数据加载时会执行 `CenterCrop(168)`、`Resize((64, 64))` 和 `ToTensor()`。

## Checkpoint 和预训练模型

本章当前不依赖第三方预训练模型。训练、重建和生成使用的是本项目代码训练出的 checkpoint。

- 默认 checkpoint：`work_dirs/vae/model.pth`
- 自检 checkpoint：`work_dirs/vae/smoke_model.pth`
- 默认输出图片：`work_dirs/vae/tmp.jpg`
- 自检输出图片：`work_dirs/vae/smoke_sample.jpg`

`--mode reconstruct` 和 `--mode generate` 需要先有 checkpoint；如果还没有训练过，请先运行 `--mode smoke` 验证链路，或运行 `--mode train` 在 CelebA 上训练。

Hugging Face 上有 MNIST VAE、Stable Diffusion VAE 等预训练模型，但它们不是本章代码的必要依赖。磁盘空间不足时不要下载大模型；后续如果章节接入第三方预训练模型，需要在这里明确模型来源、大小、下载方式和本地路径。

真实训练命令：

```powershell
uv run python -m chapters.generative_models.vae.code.main --mode train --dataset celeba --device cuda:0
uv run python -m chapters.generative_models.vae.code.main --mode reconstruct --dataset celeba --device cuda:0
uv run python -m chapters.generative_models.vae.code.main --mode generate --device cuda:0
```

checkpoint 和生成图片会写入 `work_dirs/vae/`。

## 验证命令

```powershell
uv run pytest chapters/generative_models/vae/tests -q
uv run ruff check chapters/common/utils/device.py chapters/generative_models/vae
lake build
```
