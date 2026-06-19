# ResNet — 残差网络

残差网络（Residual Network）教程。从残差连接的梯度流原理出发，覆盖
Identity Block、Convolution Block、Bottleneck 三种核心模块的 TensorFlow 实现。

> **注意**：当前实现为 TensorFlow/Keras（legacy）。PyTorch 迁移是未来工作计划。

## 目录结构

- `code/` — 规范实现
  - `code/tf_main.py` — ResNet-18 / ResNet-50 的 TensorFlow 实现（identity_block、convolution_block、bottleneck）
- `derivations/` — 完整公式推导与 Lean 形式化验证
  - `derivations/formulas.md` — 残差块公式、梯度流分析、瓶颈块推导
  - `derivations/resnet.lean` — Identity shortcut 恒等映射、梯度非零、通道匹配
- `tests/` — CPU 烟雾测试（TensorFlow 不可用时自动跳过）

## 运行命令

### 烟雾测试

```bash
uv run pytest chapters/cv/resnet/tests -q
```

### 完整训练（需要猫狗数据集 + TensorFlow 环境）

```bash
uv run python -m chapters.cv.resnet.code.tf_main
```

### 代码入口

```bash
uv run python -c "
from chapters.cv.resnet.code.tf_main import init_model
print('ResNet imported')
"
```

## 数据与依赖

- 完整训练需要猫狗数据集，放在 `data/archive/dataset/` 下。
- 数据加载复用 `chapters.cv.basic_cnn.code.dataset`。
- 烟雾测试在无 TensorFlow 时自动跳过（`pytest.importorskip`）。
- TensorFlow 环境需单独配置（不在默认 `uv sync` 依赖中）。

## 输出位置

- 训练日志输出至 stdout。
- 无检查点保存（教学实现）。

## 参考资料

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027).
- 完整公式推导：[derivations/formulas.md](derivations/formulas.md)
