# Deep Network — 深层全连接网络

多层 NumPy 神经网络教程。从浅层网络推广到任意深度，通过列表化参数管理
和循环化前向/反向传播，展示现代深度学习框架的底层设计思想。

## 目录结构

- `code/` — 规范实现
  - `code/model.py` — 模型类（`BaseRegressionModel`、`DeepNetwork`）与 `train` 函数
  - `code/dataset.py` — 猫狗图片数据集加载与预处理
  - `code/main.py` — 完整训练入口（CLI）
- `derivations/` — 完整公式推导与 Lean 形式化验证
  - `derivations/formulas.md` — 前向传播、反向传播的系统化公式推导
  - `derivations/deep_network.lean` — 梯度下降符号、矩阵形状规则等恒等式检查
- `tests/` — CPU 烟雾测试

## 运行命令

### 烟雾测试

```bash
uv run pytest chapters/fundamentals/deep_network/tests -q
```

### 完整训练（需要猫狗数据集）

```bash
uv run python -m chapters.fundamentals.deep_network.code.main
```

### 代码入口

```bash
uv run python -c "
from chapters.fundamentals.deep_network.code.model import DeepNetwork, train
print('DeepNetwork imported')
"
```

## 数据与依赖

- 完整训练需要猫狗图片数据集，放在 `data/archive/dataset/` 下。
- 烟雾测试仅依赖 NumPy，无需外部数据。
- 依赖 `numpy`、`opencv-python`（仅完整训练时需要 cv2）。

## 输出位置

- 模型检查点：`work_dirs/deep_network/model.npz`
- 训练输出目录：`work_dirs/deep_network/`

## 参考资料

- Cybenko, G. (1989). [Approximation by Superpositions of a Sigmoidal Function](https://link.springer.com/article/10.1007/BF02551274).
- Hornik, K., Stinchcombe, M., & White, H. (1989). [Multilayer Feedforward Networks are Universal Approximators](https://www.sciencedirect.com/science/article/pii/0893608089900208).
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). [Deep Learning](https://www.nature.com/articles/nature14539).
- 周弈帆（2022-07-09）。[吴恩达深度学习专项笔记（四）：深层神经网络](https://zhouyifan.net/2022/07/09/DLS-note-4/)。
- 完整公式推导：[derivations/formulas.md](derivations/formulas.md)
