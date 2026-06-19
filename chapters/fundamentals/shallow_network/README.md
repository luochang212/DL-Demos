# Shallow Network — 浅层神经网络

二分类浅层神经网络教程。从 Logistic Regression 扩展到含一个隐藏层的两
层网络，覆盖 ReLU 激活、反向传播的链式法则分解、以及隐藏层宽度对决策
边界的影响。

## 目录结构

- `code/` — 规范实现
  - `code/model.py` — 模型类（`BaseRegressionModel`、`LogisticRegression`、`ShallowNetwork`）
  - `code/generate_points.py` — 合成花瓣形二分类数据集生成与可视化
  - `code/points_classification.py` — Logistic Regression vs ShallowNetwork（2/4/10 隐藏单元）对比入口
  - `code/plot_activation_func.py` — 激活函数（Sigmoid、Tanh、ReLU、Leaky ReLU）可视化
- `derivations/` — 完整公式推导与 Lean 形式化验证
  - `derivations/formulas.md` — 网络结构、激活函数、反向传播、梯度下降的完整符号推导
  - `derivations/shallow_network.lean` — ReLU 导数、零初始化对称性等恒等式检查
- `tests/` — CPU 烟雾测试

## 运行命令

### 烟雾测试

```bash
uv run pytest chapters/fundamentals/shallow_network/tests -q
```

### 完整实验（合成数据，无需外部数据集）

```bash
uv run python -m chapters.fundamentals.shallow_network.code.points_classification
```

### 激活函数可视化

```bash
uv run python -m chapters.fundamentals.shallow_network.code.plot_activation_func
```

### 代码入口

```bash
uv run python -c "
from chapters.fundamentals.shallow_network.code.model import ShallowNetwork, train_model
print('ShallowNetwork imported')
"
```

## 数据与依赖

- 使用合成花瓣形数据（`generate_points.py` 内联生成），无需外部数据集。
- 烟雾测试仅依赖 NumPy。
- 完整实验依赖 `numpy`、`matplotlib`。

## 输出位置

- 可视化图片输出到 `work_dirs/shallow_network/`。

## 参考资料

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0).
- Nair, V., & Hinton, G. E. (2010). [Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf).
- 周弈帆（2022-06-12）。[吴恩达深度学习专项笔记（三）：浅层神经网络](https://zhouyifan.net/2022/06/12/DLS-note-3/)。
- 完整公式推导：[derivations/formulas.md](derivations/formulas.md)
