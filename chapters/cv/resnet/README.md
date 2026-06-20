# ResNet — 残差网络

残差网络（Residual Network）教程。从残差连接的梯度流原理出发，覆盖
BasicBlock、Bottleneck 两种核心模块，支持 ResNet-18/34/50/101/152。

## 目录结构

- `code/` — 规范实现
  - `code/model.py` — PyTorch ResNet 模型（BasicBlock, Bottleneck, ResNet）
  - `code/main.py` — 训练/评估 CLI（支持烟雾模式和完整训练）
  - `code/tf_main.py` — 遗留 TensorFlow/Keras 参考实现
- `derivations/` — 完整公式推导与 Lean 形式化验证
  - `derivations/formulas.md` — 残差块公式、梯度流分析、瓶颈块推导
  - `derivations/resnet.lean` — Identity shortcut 恒等映射、梯度非零、通道匹配
- `tests/` — CPU 烟雾测试（PyTorch）

## 运行命令

### 烟雾测试

```bash
uv run pytest chapters/cv/resnet/tests -q
```

### 烟雾模式训练（合成数据，2 个 epoch）

```bash
uv run python -m chapters.cv.resnet.code.main --smoke
```

### 完整训练（需要猫狗数据集）

```bash
# ResNet-18
uv run python -m chapters.cv.resnet.code.main --model resnet18 --epochs 20

# ResNet-50（带 torchvision 预训练权重）
uv run python -m chapters.cv.resnet.code.main --model resnet50 --epochs 20 --pretrained
```

## 数据与依赖

- 完整训练需要猫狗数据集，放在 `data/archive/dataset/` 下。
- 数据加载复用 `chapters.cv.basic_cnn.code.dataset`。
- 可选依赖 `torchvision` 用于预训练权重加载（`--pretrained` 标志）。
- TensorFlow 遗留代码 (`code/tf_main.py`) 需单独配置 TF 环境。

## 输出位置

- 检查点保存至 `work_dirs/resnet/`。
- 训练日志输出至 stdout。

## 参考资料

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027).
- 周弈帆（2022-08-09）。[ResNet 论文概览与精读](https://zhouyifan.net/2022/08/09/20220807-ResNet/)。
- 完整公式推导：[derivations/formulas.md](derivations/formulas.md)
