# PixelCNN 章节

本目录是 PixelCNN 教程章节的源文件目录。网页教程、可运行代码、公式推导和测试都应与这里保持一致。

## 目录结构

- `code/`：训练、采样、MNIST 数据加载和 PixelCNN / Gated PixelCNN 实现。
- `derivations/`：完整公式推导和 Lean4 公式校验代码。
- `tests/`：章节级 CPU smoke tests。

## 运行命令

不依赖 MNIST 的真实 CLI 自检：

```powershell
uv run python -m chapters.generative_models.pixelcnn.code.main --mode smoke --device cpu
```

该命令会使用合成 28x28 灰度图片跑一次真实训练路径，保存 checkpoint，并生成一张样例图。
`--device auto` 会优先使用 CUDA，其次使用 macOS MPS，最后回退到 CPU；没有 GPU 也可以用 CPU 跑通自检。

## 数据准备

PixelCNN 训练使用 MNIST 手写数字图片。

- 数据来源：`torchvision.datasets.MNIST`
- 官方来源：[The MNIST Database](http://yann.lecun.com/exdb/mnist/)
- 默认路径：`data/mnist/`
- 数据规模：60,000 张训练图片，10,000 张测试图片，灰度 28x28。
- 用途：快速训练和观察逐像素自回归生成效果。

快速训练和采样：

```powershell
uv run python -m chapters.generative_models.pixelcnn.code.main --mode train --epochs 1 --batch-size 128 --model-id 1 --color-level 8 --device auto --checkpoint work_dirs/pixelcnn/mnist_model.pth
uv run python -m chapters.generative_models.pixelcnn.code.main --mode sample --model-id 1 --color-level 8 --device auto --checkpoint work_dirs/pixelcnn/mnist_model.pth --output work_dirs/pixelcnn/mnist_sample.jpg --n-sample 16
```

## Checkpoint 和预训练模型

本章当前不依赖第三方预训练模型。训练和采样使用的是本项目代码训练出的 checkpoint。

- 默认 checkpoint：`work_dirs/pixelcnn/model_1_8.pth`
- 自检 checkpoint：`work_dirs/pixelcnn/smoke_model.pth`
- 默认输出图片：`work_dirs/pixelcnn/pixelcnn_1_8.jpg`
- 自检输出图片：`work_dirs/pixelcnn/smoke_sample.jpg`

`--mode sample` 需要先有 checkpoint；如果还没有训练过，请先运行 `--mode smoke` 验证链路，或运行 `--mode train` 在 MNIST 上训练。

## 验证命令

```powershell
uv run pytest chapters/generative_models/pixelcnn/tests -q
uv run ruff check chapters/generative_models/pixelcnn
lake build
```
