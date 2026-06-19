# Logistic Regression — 二分类起点

二分类逻辑回归教程。从线性打分、Sigmoid 激活、二元交叉熵到梯度下降，
完整覆盖「最小的可训练神经网络」。

## 运行命令

### 烟雾测试

```bash
uv run pytest chapters/fundamentals/logistic_regression/tests -q
```

### 完整训练（需要猫狗数据集）

```bash
uv run python -m chapters.fundamentals.logistic_regression.code.main
```

### 代码入口

```bash
uv run python -c "
from chapters.fundamentals.logistic_regression.code.model import sigmoid, train_step
print('Logistic regression core functions imported')
"
```

## 数据与依赖

- 完整训练需要猫狗图片数据集，放在 `data/archive/dataset/` 下。
- 烟雾测试仅依赖 NumPy，无需外部数据。
- 依赖 `numpy`、`opencv-python`（仅完整训练时需要 cv2）。

## 输出位置

- 无持久输出。训练结果直接打印到终端。

## 参考资料

- Cox, D. R. (1958). [The Regression Analysis of Binary Sequences](https://www.jstor.org/stable/2983890).
- 周弈帆（2022-05-10）。[吴恩达深度学习专项笔记（二）：逻辑回归](https://zhouyifan.net/2022/05/10/DLS-note-2/)。
- 周弈帆（2022-10-13）。[从零理解熵、交叉熵、KL散度](https://zhouyifan.net/2022/10/13/20221012-entropy/)。
- 完整公式推导：[derivations/formulas.md](derivations/formulas.md)
