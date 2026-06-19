# Regularization 公式推导

网页教程负责保持可读性；这里补上 Weight Decay（L2 正则化）和 Dropout
的完整符号推导。

## 1. 符号约定

| 符号 | 含义 | 代码变量 |
|------|------|---------|
| $W^{(l)}$ | 第 $l$ 层权重矩阵 | `self.W[i]` |
| $m$ | 批量大小 | `self.m` |
| $\lambda$ | L2 正则化系数 | `LAMBDA` (代码中常数 4) |
| $p$ | Dropout 保留概率 | `keep_prob` (代码中 0.5) |
| $d$ | Dropout 二值掩码 | `d` |
| $\mathcal{L}$ | 原始损失（BCE） | 基础 loss |
| $\mathcal{L}_{\text{reg}}$ | 正则化后总损失 | `model.loss(...)` |

## 2. L2 正则化 / Weight Decay

### 2.1 损失函数

L2 正则化在损失中加入所有权重的平方和：

$$\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{BCE}} + \frac{\lambda}{2m} \sum_{l} \|W^{(l)}\|_F^2$$

其中 $\|W\|_F^2 = \sum_{i,j} W_{ij}^2$ 是 Frobenius 范数的平方。

对应代码（`code/model.py`）：

```python
LAMBDA = 4
tot = np.mean(-(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)))
for i in range(self.num_layer):
    tot += np.sum(self.W[i] * self.W[i]) * LAMBDA / 2 / self.m
return tot
```

### 2.2 梯度

对 $W^{(l)}$ 求导：

$$\frac{\partial \mathcal{L}_{\text{reg}}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}_{\text{BCE}}}{\partial W^{(l)}} + \frac{\lambda}{m} W^{(l)}$$

### 2.3 参数更新

梯度下降 + weight decay 的结合更新：

$$W^{(l)} \leftarrow W^{(l)} - \eta \left( \frac{\partial \mathcal{L}_{\text{BCE}}}{\partial W^{(l)}} + \frac{\lambda}{m} W^{(l)} \right)$$

$$= \left(1 - \frac{\eta \lambda}{m}\right) W^{(l)} - \eta \cdot \frac{\partial \mathcal{L}_{\text{BCE}}}{\partial W^{(l)}}$$

对应代码：

```python
self.W[i] = (1 - learning_rate * LAMBDA / self.m) * self.W[i] - learning_rate * self.dW_cache[i]
```

权重在每一步被衰减因子 $(1 - \eta\lambda/m)$ 缩放，因此称为 "weight decay"。

### 2.4 为什么 Weight Decay 防过拟合

- 大权重 → 模型对输入的微小变化敏感 → 决策边界剧烈弯曲 → 过拟合
- L2 惩罚迫使权重接近零 → 模型更平滑 → 泛化更好
- 等价于对权重施加高斯先验 $\mathcal{N}(0, 1/\lambda)$ 的 MAP 估计

## 3. Dropout

### 3.1 Inverted Dropout 前向传播

标准 Dropout 在训练时以概率 $1-p$ 随机丢弃神经元：

$$A_{\text{dropout}} = A \odot d$$

其中 $d \sim \text{Bernoulli}(p)$ 是二值掩码，$\odot$ 表示逐元素乘法。

**Inverted Dropout** 在丢弃后除以 $p$ 以保持期望不变：

$$A_{\text{dropout}} = A \odot d \cdot \frac{1}{p}$$

这样测试时无需缩放，直接使用完整网络。

对应代码：

```python
keep_prob = 0.5
d = np.random.rand(*A.shape) < keep_prob
A = A * d / keep_prob
self.dropout_mask_cache[i] = d
```

### 3.2 期望保持

训练时：

$$\mathbb{E}[A \cdot d / p] = A \cdot \mathbb{E}[d] / p = A \cdot p / p = A$$

因此 Dropout 层的输出期望等于原始激活值。

### 3.3 反向传播

Dropout 掩码在反向传播中同样需要应用：

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial A_{\text{dropout}}} \odot d \cdot \frac{1}{p}$$

对应代码：

```python
if self.dropout:
    keep_prob = 0.5
    dA = dA * self.dropout_mask_cache[i] / keep_prob
```

### 3.4 为什么 Dropout 防过拟合

- **打破神经元共适应**：每个神经元不能依赖特定其他神经元的存在
- **隐式集成**：每次前向相当于从 $2^n$ 个子网络中随机采样一个，测试时近似几何平均
- Srivastava et al. (2014) 证明 Dropout 等价于对权重施加自适应 L2 正则化

## 4. 实验对比

| 正则化 | 策略 | 效果 |
|--------|------|------|
| None | 无正则化 | 过拟合，决策边界过度弯曲 |
| Weight Decay ($\lambda=4$) | L2 惩罚 | 权重接近零，边界平滑 |
| Dropout ($p=0.5$) | 随机丢弃 | 打破共适应，边界平滑 |

## Lean 已验证的恒等式

`derivations/regularization.lean` 检查以下代码级代数恒等式：

- Weight decay 更新公式的代数等价形式
- Dropout 期望保持：$\mathbb{E}[d]/p = 1$（以整数恒等式编码）
- 权重衰减因子展开

Lean 文件不试图形式化概率论（Bernoulli 分布、期望等）或泛化理论。
它只检查代码中正则化更新步骤的关键整数比例关系。
