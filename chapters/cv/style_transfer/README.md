# Style Transfer — 神经风格迁移

Gatys et al. 神经风格迁移教程。将内容图像与风格图像分离，通过优化生成
图像的 VGG 特征来同时匹配内容和风格。

## 目录结构

- `code/` — 规范实现
  - `code/style_transfer.py` — 完整风格迁移（`ContentLoss`、`StyleLoss`、`Normalization`、`run_style_transfer`）
  - `code/combine_img.py` — 内容+风格联合优化演示
  - `code/copy_img.py` — 纯风格复制演示
- `derivations/` — 完整公式推导与 Lean 形式化验证
  - `derivations/formulas.md` — Content Loss、Gram Matrix、Style Loss、Total Loss 的完整符号推导
  - `derivations/style_transfer.lean` — 损失结合律、单位权重恒等式
- `tests/` — CPU 烟雾测试
- `dancing.jpg`, `picasso.jpg` — 默认内容/风格图像

## 运行命令

### 烟雾测试

```bash
uv run pytest chapters/cv/style_transfer/tests -q
```

### 完整运行

```bash
uv run python -m chapters.cv.style_transfer.code.style_transfer \
  --content chapters/cv/style_transfer/dancing.jpg \
  --style chapters/cv/style_transfer/picasso.jpg \
  --output work_dirs/style-transfer.jpg
```

首次运行会自动下载 torchvision 的 VGG19 预训练权重。

### 代码入口

```bash
uv run python -c "
from chapters.cv.style_transfer.code.style_transfer import run_style_transfer, gram
print('Style Transfer imported')
"
```

## 数据与依赖

- 使用内置 `dancing.jpg` 和 `picasso.jpg` 作为默认内容/风格图像。
- 依赖 `torch`、`torchvision`（VGG19 预训练权重）。
- 烟雾测试不加载 VGG 权重。

## 输出位置

- 风格迁移结果：`work_dirs/style-transfer.jpg`（默认路径）。

## 参考资料

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576).
- Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html).
- 周弈帆（2022-05-31）。[Neural Style Transfer 风格迁移经典论文讲解与 PyTorch 实现](https://zhouyifan.net/2022/05/31/20220531-styletransfer/)。
- 完整公式推导：[derivations/formulas.md](derivations/formulas.md)
