# Basic CNN — CNN 基础：从卷积运算到图像分类

NumPy 手写 2D 卷积（前向+反向）与 PyTorch 猫狗分类实战教程。
从卷积的滑动窗口、填充、步幅、膨胀等底层细节出发，到完整的 CNN 分类网络训练。

## 目录结构

- `code/` — 规范实现
  - `code/dataset.py` — 猫狗图片数据集加载（`get_cat_set`、`load_set`）
  - `code/np_conv.py` — NumPy 2D 卷积前向（支持 stride/padding/dilation/groups）
  - `code/np_conv_backward.py` — NumPy 2D 卷积前向+反向（含梯度传播）
  - `code/pt_main.py` — PyTorch CNN 猫狗分类（完整训练+评估）
  - `code/tf_main.py` — TensorFlow CNN 猫狗分类（legacy 对照实现）
- `derivations/` — 完整公式推导与 Lean 形式化验证
  - `derivations/formulas.md` — 卷积前向/反向的符号推导、输出尺寸计算、梯度累加
  - `derivations/basic_cnn.lean` — 输出尺寸恒等式、梯度累加交换性、分组通道恒等
- `tests/` — CPU 烟雾测试

## 运行命令

### 烟雾测试

```bash
uv run pytest chapters/cv/basic_cnn/tests -q
```

### 完整训练（需要猫狗数据集）

PyTorch 版本：

```bash
uv run python -m chapters.cv.basic_cnn.code.pt_main
```

TensorFlow 版本（需要独立 TF 环境）：

```bash
uv run python -m chapters.cv.basic_cnn.code.tf_main
```

### 代码入口

```bash
uv run python -c "
from chapters.cv.basic_cnn.code.np_conv import conv2d
from chapters.cv.basic_cnn.code.pt_main import init_model
print('Basic CNN imported')
"
```

## 数据与依赖

- 完整训练需要猫狗数据集，从 [Kaggle](https://www.kaggle.com/datasets/fusicfenta/cat-and-dog) 下载后放在 `data/archive/dataset/` 下。
- 烟雾测试仅依赖 PyTorch，无需外部数据。
- NumPy 卷积的 inline 测试依赖 PyTorch（用作 reference 验证）。
- TensorFlow 版本保留为 legacy 对照，需要独立 TF 环境。

## 输出位置

- PyTorch 训练无检查点保存（教学实现）。
- 训练日志输出至 stdout。

## 参考资料

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). [Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791).
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html).
- Ioffe, S., & Szegedy, C. (2015). [Batch Normalization](https://arxiv.org/abs/1502.03167).
- CS231n: [Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/).
- 完整公式推导：[derivations/formulas.md](derivations/formulas.md)
