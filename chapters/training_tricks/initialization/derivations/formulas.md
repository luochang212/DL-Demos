# Initialization 公式推导

网页教程负责保持可读性；这里补上权重初始化策略的完整符号推导。
推导聚焦 He 初始化（Kaiming 初始化）的前向方差分析，以及与 Xavier 初始化的对比。

## 1. 符号约定

| 符号 | 含义 | 代码变量 |
|------|------|---------|
| $n_l$ | 第 $l$ 层输入维度 | `neuron_cnt[l]` |
| $n_{l+1}$ | 第 $l$ 层输出维度 | `neuron_cnt[l+1]` |
| $W^{(l)}$ | 第 $l$ 层权重矩阵，形状 $(n_{l+1}, n_l)$ | `self.W[i]` |
| $w^{(l)}_{ij}$ | $W^{(l)}$ 的单个元素 | `self.W[i][j,k]` |
| $x^{(l)}$ | 第 $l$ 层输入向量 | `A` (forward 中的中间激活) |
| $y^{(l)}$ | 第 $l$ 层线性输出（激活前） | `Z` |
| $\text{Var}(x)$ | 随机变量 $x$ 的方差 | — |

所有初始化策略共享偏置初始化 $b = 0$，因此方差分析聚焦权重。

## 2. 问题的根源：方差爆炸与消失

考虑一层全连接变换：

$$y_j^{(l)} = \sum_{i=1}^{n_l} w_{ji}^{(l)} x_i^{(l)}$$

假设：
- $w_{ji}$ 与 $x_i$ 独立
- $w_{ji}$ 均值为 0，方差为 $\text{Var}(w)$
- $x_i$ 均值为 0，方差为 $\text{Var}(x)$

则 $y_j$ 的方差为：

$$\text{Var}(y_j) = \sum_{i=1}^{n_l} \text{Var}(w_{ji} x_i)$$

由独立性：

$$\text{Var}(w_{ji} x_i) = \text{Var}(w_{ji}) \cdot \text{Var}(x_i) + \text{Var}(w_{ji})\mathbb{E}[x_i]^2 + \text{Var}(x_i)\mathbb{E}[w_{ji}]^2$$

利用零均值假设，后两项消去：

$$\text{Var}(y_j) = n_l \cdot \text{Var}(w) \cdot \text{Var}(x)$$

这意味着每经过一层，信号方差被放大 $n_l \cdot \text{Var}(w)$ 倍。
若 $\text{Var}(w) \neq 1/n_l$，经过多层后方差会指数级爆炸或消失。

## 3. Xavier / Glorot 初始化

Xavier 初始化（Glorot & Bengio, 2010）的目标是保持前向与反向传播方差不变。
对于前向传播，要求 $\text{Var}(y) = \text{Var}(x)$，即：

$$n_l \cdot \text{Var}(w) = 1 \quad\Rightarrow\quad \text{Var}(w) = \frac{1}{n_l}$$

对于反向传播，梯度从 $n_{l+1}$ 维传播回 $n_l$ 维，对称地要求 $\text{Var}(w) = 1 / n_{l+1}$。

Xavier 取两者的调和平均：

$$\text{Var}(w) = \frac{2}{n_l + n_{l+1}}$$

对于均匀分布 $U(-a, a)$，其方差为 $a^2/3$，因此：

$$a = \sqrt{\frac{6}{n_l + n_{l+1}}}$$

对应正态分布 $\mathcal{N}(0, \sigma^2)$：

$$\sigma = \sqrt{\frac{2}{n_l + n_{l+1}}}$$

**Xavier 的局限**：推导假设激活函数是线性的（或近似线性的，如 tanh），
对 ReLU 等非对称激活函数不成立。

## 4. He / Kaiming 初始化

He 初始化（He et al., 2015）专门针对 ReLU 激活函数设计。
ReLU 将一半的输入置零：$\text{ReLU}(x) = \max(0, x)$。

对于线性输出 $y = Wx$ 后再经 ReLU：

$$\text{Var}(\text{ReLU}(y_j))$$

假设 $w$ 服从零均值对称分布，则 $y_j$ 也服从零均值对称分布。
经过 ReLU 后：

$$\mathbb{E}[\text{ReLU}(y)^2] = \frac{1}{2} \mathbb{E}[y^2]$$

因此：

$$\text{Var}(\text{ReLU}(y)) = \frac{1}{2} \text{Var}(y)$$

前向方差传播要求 $\text{Var}(x^{(l+1)}) = \text{Var}(x^{(l)})$：

$$\frac{1}{2} \cdot n_l \cdot \text{Var}(w) \cdot \text{Var}(x^{(l)}) = \text{Var}(x^{(l)})$$

$$\Rightarrow \text{Var}(w) = \frac{2}{n_l}$$

对于正态分布：

$$\sigma = \sqrt{\frac{2}{n_l}}$$

对应代码（`code/model.py`）：

```python
self.W.append(
    np.random.randn(neuron_cnt[i + 1], neuron_cnt[i])
    * np.sqrt(2 / neuron_cnt[i])
)
```

`np.random.randn` 产生 $\mathcal{N}(0, 1)$，乘以 $\sqrt{2/n_l}$ 后得到 $\mathcal{N}(0, 2/n_l)$，
符合 He 初始化的方差要求。

## 5. 零初始化与对称性问题

若所有权重初始化为零：

$$W = 0$$

则前向传播中 $y = Wx + b = 0 + 0 = 0$（$b$ 也初始化为 0）。
激活后 $\text{ReLU}(0) = 0$，所有隐藏单元输出相同。

反向传播时：

$$dW = \frac{1}{m} dZ \cdot A^T$$

由于所有隐藏单元 $A$ 相同，每一层的所有 $dW_{ji}$ 也相同。
梯度下降后，该层所有神经元依然完全相同 — 这就是**对称性问题**。
无论训练多少步，网络等价于只有一个神经元。

代码中零初始化的表现（见 `main.py` 实验结果）：
loss 几乎不下降，决策边界与随机猜测无异。

## 6. 随机初始化方差过大问题

代码中「随机初始化」使用 $\sigma = 5$：

```python
self.W.append(np.random.randn(neuron_cnt[i + 1], neuron_cnt[i]) * 5)
```

此时 $\text{Var}(w) = 25$，每层信号放大 $n_l \cdot 25$。
对于 $n_l = 10$ 的层，方差放大 250 倍。经过 2-3 层后，
激活值进入 Sigmoid 的饱和区，梯度趋近于零（梯度消失）。

## 7. 实验对比

| 初始化 | 方差 | 效果 |
|--------|------|------|
| Zero | 0 | 对称性破坏失败，等价于线性模型 |
| Random ($\sigma = 5$) | 25 | 方差爆炸 → 梯度消失 |
| He | $2/n_l$ | 方差稳定传播，快速收敛 |

## Lean 已验证的恒等式

`derivations/initialization.lean` 检查以下代码级代数恒等式：

- He 初始化的方差公式：`2 / n_l` 的代数等价形式
- 零初始化时前向传播输出恒为零
- 零初始化时梯度对所有神经元完全相同（对称性不破坏）

Lean 文件不试图形式化概率论（方差、正态分布等）或神经网络训练收敛性。
它只检查代码中实现初始化标准差计算时使用的关键整数比例关系。
