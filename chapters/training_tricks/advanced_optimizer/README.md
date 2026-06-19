# Advanced Optimizer — 高级优化器

从零实现 SGD、Momentum、RMSProp、Adam 四种优化器的教程。对比不同优化器
在猫狗分类任务上的收敛速度与稳定性，推导每种优化器的数学原理。

## 目录结构

- `code/` — 规范实现
  - `code/optimizer.py` — `GradientDescent`、`Momentum`、`RMSProp`、`Adam` 优化器（含偏差修正与 LR 调度）
  - `code/model.py` — `DeepNetwork` 模型类（He 初始化，支持小批量训练与检查点恢复）
  - `code/main.py` — 完整训练入口（猫狗数据集，四种优化器对比）
  - `code/single_step.py` — 单步训练脚本（快速验证）
  - `code/compare_optimizer.py` — 优化器对比曲线（硬编码结果，参考用）
- `derivations/` — 完整公式推导与 Lean 形式化验证
  - `derivations/formulas.md` — SGD/Momentum/RMSProp/Adam 更新规则的完整符号推导
  - `derivations/advanced_optimizer.lean` — Adam 偏差修正、Momentum 递推等恒等式检查
- `tests/` — CPU 烟雾测试

## 运行命令

### 烟雾测试

```bash
uv run pytest chapters/training_tricks/advanced_optimizer/tests -q
```

### 完整训练（需要猫狗数据集）

```bash
uv run python -m chapters.training_tricks.advanced_optimizer.code.main
```

### 代码入口

```bash
uv run python -c "
from chapters.training_tricks.advanced_optimizer.code.optimizer import Adam, Momentum
print('Adam and Momentum imported')
"
```

## 数据与依赖

- 完整训练需要猫狗图片数据集，放在 `data/archive/dataset/` 下。
- 烟雾测试仅依赖 NumPy，无需外部数据。
- 依赖 `numpy`、`matplotlib`、`opencv-python`（仅完整训练时需要 cv2）。

## 输出位置

- 模型检查点：`work_dirs/advanced_optimizer/model_latest.npz`
- 训练输出目录：`work_dirs/advanced_optimizer/`

## 参考资料

- Kingma, D. P., & Ba, J. (2015). [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
- Ruder, S. (2017). [An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747).
- 完整公式推导：[derivations/formulas.md](derivations/formulas.md)
