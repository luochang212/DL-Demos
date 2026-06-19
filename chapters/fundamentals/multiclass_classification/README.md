# Multiclass Classification — 多分类

PyTorch 多分类教程。从二分类推广到 C 类，覆盖 Softmax 函数、交叉熵损失、
以及 logits 与概率的区别。

## 目录结构

- `code/` — 规范实现
  - `code/model.py` — `MulticlassClassificationNet` 模型类与 `train` 函数
  - `code/points_classification.py` — 合成三类数据集（二次边界）生成与可视化
  - `code/main.py` — PyTorch 完整训练入口
- `derivations/` — 完整公式推导与 Lean 形式化验证
  - `derivations/formulas.md` — Softmax、交叉熵、梯度推导的完整符号推导
  - `derivations/multiclass_classification.lean` — Softmax-CE 梯度、One-Hot 求和等恒等式检查
- `tests/` — CPU 烟雾测试
- `tf_main.py` — Legacy TensorFlow 实现（参考用，需要独立 TF 环境）

## 运行命令

### 烟雾测试

```bash
uv run pytest chapters/fundamentals/multiclass_classification/tests -q
```

### 完整实验（PyTorch，合成数据，无需外部数据集）

```bash
uv run python -m chapters.fundamentals.multiclass_classification.code.main
```

### 代码入口

```bash
uv run python -c "
from chapters.fundamentals.multiclass_classification.code.model import MulticlassClassificationNet
print('MulticlassClassificationNet imported')
"
```

## 数据与依赖

- 使用合成三类数据（`points_classification.py` 内联生成），无需外部数据集。
- 烟雾测试与完整实验依赖 `torch`、`numpy`、`matplotlib`。

## 输出位置

- 可视化图片输出到 `work_dirs/multiclass_classification/`。

## 参考资料

- Bridle, J. S. (1990). [Probabilistic Interpretation of Feedforward Classification Network Outputs](https://link.springer.com/chapter/10.1007/978-1-4471-2038-4_24).
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). [Deep Learning](https://www.deeplearningbook.org/). MIT Press.
- 周弈帆（2022-08-03）。[吴恩达深度学习专项笔记（五）：多分类问题](https://zhouyifan.net/2022/08/03/DLS-note-5/)。
- 完整公式推导：[derivations/formulas.md](derivations/formulas.md)
