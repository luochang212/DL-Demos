# Regularization — 正则化

神经网络正则化策略教程。对比无正则化、Weight Decay（L2）、Dropout 三种
策略在过参数化网络上的过拟合抑制效果，从数学角度推导两种正则化的原理。

## 目录结构

- `code/` — 规范实现
  - `code/model.py` — `DeepNetwork` 模型类（支持 none/weight decay/dropout 三种策略）与 `train` 函数
  - `code/points_classification.py` — 带噪声线性二分类合成数据集生成与可视化
  - `code/main.py` — 完整实验入口（三种正则化对比）
- `derivations/` — 完整公式推导与 Lean 形式化验证
  - `derivations/formulas.md` — L2 正则化梯度推导、Inverted Dropout 期望保持、反向传播
  - `derivations/regularization.lean` — Weight Decay 更新公式、Dropout 期望恒等式检查
- `tests/` — CPU 烟雾测试

## 运行命令

### 烟雾测试

```bash
uv run pytest chapters/training_tricks/regularization/tests -q
```

### 完整实验（合成数据，无需外部数据集）

```bash
uv run python -m chapters.training_tricks.regularization.code.main
```

### 代码入口

```bash
uv run python -c "
from chapters.training_tricks.regularization.code.model import DeepNetwork, train
print('DeepNetwork (regularization variants) imported')
"
```

## 数据与依赖

- 使用合成线性可分数数据集（`points_classification.py` 内联生成，含 20% 噪声），无需外部数据。
- 依赖 `numpy`、`matplotlib`。

## 输出位置

- 可视化图片输出通过 `points_classification.py` 中的 `visualize` 函数。

## 参考资料

- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html).
- Loshchilov, I., & Hutter, F. (2019). [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
- 完整公式推导：[derivations/formulas.md](derivations/formulas.md)
