# 深层神经网络 — 完整公式推导

## 1. 网络结构

$L$ 层全连接网络（输入层不计入 $L$）：

**第 $l$ 层**（$1 \leq l \leq L$）：

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$

- $A^{[0]} = X \in \mathbb{R}^{n_x \times m}$：输入
- $A^{[L]} = \hat{Y} \in \mathbb{R}^{1 \times m}$：输出
- $W^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$，$b^{[l]} \in \mathbb{R}^{n^{[l]} \times 1}$

## 2. 前向传播（系统化）

```python
A = X
cache A[0]
for l in 1..L:
    Z[l] = W[l] @ A + b[l]      # 线性变换
    A = g[l](Z[l])               # 激活
    cache Z[l], A
```

## 3. 损失函数

二分类使用二元交叉熵：

$$J = -\frac{1}{m} \sum_{i=1}^m \left[y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)\right]$$

## 4. 反向传播（系统化）

### 输出层起始梯度

$$\frac{\partial J}{\partial A^{[L]}} = -\frac{Y}{A^{[L]}} + \frac{1-Y}{1-A^{[L]}}$$

### 逐层回传（$l = L, L-1, \ldots, 1$）

**步骤 1：通过激活函数**

$$\frac{\partial J}{\partial Z^{[l]}} = \frac{\partial J}{\partial A^{[l]}} \odot g^{[l]'}(Z^{[l]})$$

**步骤 2：对参数求导**

$$\frac{\partial J}{\partial W^{[l]}} = \frac{1}{m} \frac{\partial J}{\partial Z^{[l]}} A^{[l-1]T}$$

$$\frac{\partial J}{\partial b^{[l]}} = \frac{1}{m} \sum_{i=1}^m \frac{\partial J}{\partial Z^{[l](i)}}$$

**步骤 3：回传至上一层**

$$\frac{\partial J}{\partial A^{[l-1]}} = W^{[l]T} \frac{\partial J}{\partial Z^{[l]}}$$

### 矩阵形状验证

| 变量 | 形状 |
|---|---|
| $dZ^{[l]}$ | $(n^{[l]}, m)$ |
| $dW^{[l]}$ | $(n^{[l]}, n^{[l-1]})$ |
| $db^{[l]}$ | $(n^{[l]}, 1)$ |
| $dA^{[l-1]}$ | $(n^{[l-1]}, m)$ |

梯度张量与其对应的参数/输出形状相同。

## 5. 梯度下降更新

对 $l = 1, 2, \ldots, L$：

$$W^{[l]} \leftarrow W^{[l]} - \alpha \frac{\partial J}{\partial W^{[l]}}$$
$$b^{[l]} \leftarrow b^{[l]} - \alpha \frac{\partial J}{\partial b^{[l]}}$$

## 6. 深层 vs 浅层：计算复杂度直觉

拟合 $n$ 个数相加 $x_1 + x_2 + \cdots + x_n$：

- **深层**（$\log_2 n$ 层二叉树）：每层做两两相加，共 $n-1$ 个加法单元
- **浅层**（单层）：需要枚举 $O(2^n)$ 种组合，每个神经元对应一种组合

深层以更少的参数实现更强的表达能力。这是深层网络的**层次化组合**优势——逐层构建更高级的抽象，而非在同一层穷举。
