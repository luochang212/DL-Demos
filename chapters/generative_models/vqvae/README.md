# VQVAE 章节

本目录是 VQVAE 教程章节的源文件目录。网页教程、可运行代码、公式推导和测试都应与这里保持一致。

## 目录结构

- `code/`：VQ-VAE、PixelCNN 先验、数据加载、训练和采样实现。
- `derivations/`：完整公式推导和 Lean4 公式校验代码。
- `tests/`：章节级 CPU smoke tests。

## 运行命令

不依赖 MNIST 的真实 CLI 自检：

```powershell
uv run python -m chapters.generative_models.vqvae.code.main --mode smoke --device cpu
```

该命令会使用合成 28x28 灰度图片训练一步 VQ-VAE，保存 VQ-VAE checkpoint 和 PixelCNN 先验 checkpoint，并生成重建图与采样图。

## 数据准备

VQVAE 快速训练使用 MNIST；配置文件也保留 CelebA / CelebA-HQ 路径。

- MNIST 来源：`torchvision.datasets.MNIST`
- MNIST 默认路径：`data/mnist/`
- CelebA 默认路径：`data/celebA/img_align_celeba/`
- CelebA-HQ 默认路径：`data/celebA/celeba_hq_256/`

完整训练仍使用配置编号：

```powershell
uv run python -m chapters.generative_models.vqvae.code.main --mode full -c 0 --device auto
```

## Checkpoint 和预训练模型

本章当前不依赖第三方预训练模型。两阶段训练会产生两个本项目 checkpoint：

- VQ-VAE checkpoint：`work_dirs/vqvae/model.pth`
- PixelCNN 先验 checkpoint：`work_dirs/vqvae/gen_model.pth`
- 自检 VQ-VAE checkpoint：`work_dirs/vqvae/smoke_model.pth`
- 自检 PixelCNN checkpoint：`work_dirs/vqvae/smoke_gen_model.pth`

Hugging Face 上有 VQ-VAE、VQGAN 和离散 token 图像生成相关模型，但它们不是本章代码的必要依赖。磁盘空间不足时不要下载大模型；后续如果章节接入第三方预训练模型，需要在这里明确模型来源、大小、下载方式和本地路径。

## 验证命令

```powershell
uv run pytest chapters/generative_models/vqvae/tests -q
uv run ruff check chapters/generative_models/vqvae
lake build
```
