# Initialization — 参数初始化

神经网络权重初始化策略教程。对比零初始化、随机初始化、He 初始化三种
策略对训练收敛和决策边界的影响，从方差传播角度推导 He 初始化的公式。

## 目录结构

- `code/` — 规范实现
  - `code/model.py` — `DeepNetwork` 模型类（支持 zeros/random/he 三种策略）与 `train` 函数
  - `code/points_classification.py` — 环形二分类合成数据集生成与可视化
  - `code/main.py` — 完整实验入口（三种初始化对比）
- `derivations/` — 完整公式推导与 Lean 形式化验证
  - `derivations/formulas.md` — He/Xavier 初始化方差传播、零初始化对称性、方差爆炸分析
  - `derivations/initialization.lean` — He 方差公式、零初始化恒等式检查
- `tests/` — CPU 烟雾测试

## 运行命令

### 烟雾测试

```bash
uv run pytest chapters/training_tricks/initialization/tests -q
```

### 完整实验（合成数据，无需外部数据集）

```bash
uv run python -m chapters.training_tricks.initialization.code.main
```

### 代码入口

```bash
uv run python -c "
from chapters.training_tricks.initialization.code.model import DeepNetwork, train
print('DeepNetwork (initialization variants) imported')
"
```

## 数据与依赖

- 使用合成环形数据集（`points_classification.py` 内联生成），无需外部数据。
- 依赖 `numpy`、`matplotlib`。

## 输出位置

- 可视化图片输出通过 `points_classification.py` 中的 `visualize` 函数。

## 参考资料

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852).
- Glorot, X., & Bengio, Y. (2010). [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html).
- 完整公式推导：[derivations/formulas.md](derivations/formulas.md)
