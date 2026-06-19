# 浅层神经网络 — 完整公式推导

## 1. 网络结构

**单隐藏层神经网络**（输入 → 隐藏层 → 输出层）：

$$
\begin{aligned}
Z^{[1]} &= W^{[1]} X + b^{[1]} \\
A^{[1]} &= g^{[1]}(Z^{[1]}) \\
Z^{[2]} &= W^{[2]} A^{[1]} + b^{[2]} \\
A^{[2]} &= g^{[2]}(Z^{[2]}) = \hat{Y}
\end{aligned}
$$

- $X \in \mathbb{R}^{n_x \times m}$：$m$ 个样本，每个 $n_x$ 维
- $W^{[1]} \in \mathbb{R}^{n_1 \times n_x}$，$b^{[1]} \in \mathbb{R}^{n_1 \times 1}$
- $W^{[2]} \in \mathbb{R}^{1 \times n_1}$，$b^{[2]} \in \mathbb{R}^{1 \times 1}$
- $g^{[1]}$：隐藏层激活函数（ReLU）
- $g^{[2]}$：输出层激活函数（sigmoid）

## 2. 激活函数及其导数

### ReLU

$$g(z) = \max(0, z)$$

$$g'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}$$

在 $z=0$ 处约定 $g'(0) = 1$（右极限）。

### Sigmoid（见 logistic_regression 推导）

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

## 3. 损失函数

二元交叉熵（同 logistic regression）：

$$J = -\frac{1}{m} \sum_{i=1}^m \left[y_i \log \hat{y}_i + (1-y_i) \log(1 - \hat{y}_i)\right]$$

## 4. 反向传播推导

### 4.1 输出层（第 2 层）

输出层使用 sigmoid，梯度与 logistic regression 完全相同：

$$\frac{\partial J}{\partial Z^{[2]}} = A^{[2]} - Y \quad \in \mathbb{R}^{1 \times m}$$

记 $dZ^{[2]} = \frac{\partial J}{\partial Z^{[2]}}$。

对 $W^{[2]}$ 求导（链式法则）：

$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$

$$\frac{\partial Z^{[2]}}{\partial W^{[2]}} = A^{[1]T}$$

$$\frac{\partial J}{\partial W^{[2]}} = \frac{1}{m} dZ^{[2]} A^{[1]T} \quad \in \mathbb{R}^{1 \times n_1}$$

对 $b^{[2]}$ 求导：

$$\frac{\partial J}{\partial b^{[2]}} = \frac{1}{m} \sum_{i=1}^m dZ^{[2](i)} \quad \in \mathbb{R}^{1 \times 1}$$

### 4.2 隐藏层（第 1 层）

梯度从输出层回传至 $A^{[1]}$：

$$\frac{\partial J}{\partial A^{[1]}} = W^{[2]T} dZ^{[2]} \quad \in \mathbb{R}^{n_1 \times m}$$

通过激活函数的链式法则：

$$\frac{\partial J}{\partial Z^{[1]}} = \frac{\partial J}{\partial A^{[1]}} \odot g^{[1]'}(Z^{[1]})$$

其中 $\odot$ 是逐元素乘法。因 $g^{[1]}$ 为 ReLU，$g^{[1]'}(Z^{[1]})$ 在 $Z^{[1]} > 0$ 处为 $1$，$Z^{[1]} < 0$ 处为 $0$。

记 $dZ^{[1]} = \frac{\partial J}{\partial Z^{[1]}} \in \mathbb{R}^{n_1 \times m}$。

对 $W^{[1]}$ 求导：

$$Z^{[1]} = W^{[1]} X + b^{[1]}$$

$$\frac{\partial J}{\partial W^{[1]}} = \frac{1}{m} dZ^{[1]} X^T \quad \in \mathbb{R}^{n_1 \times n_x}$$

对 $b^{[1]}$ 求导：

$$\frac{\partial J}{\partial b^{[1]}} = \frac{1}{m} \sum_{i=1}^m dZ^{[1](i)} \quad \in \mathbb{R}^{n_1 \times 1}$$

## 5. 梯度下降更新

$$
\begin{aligned}
W^{[1]} &\leftarrow W^{[1]} - \alpha \frac{\partial J}{\partial W^{[1]}} \\
b^{[1]} &\leftarrow b^{[1]} - \alpha \frac{\partial J}{\partial b^{[1]}} \\
W^{[2]} &\leftarrow W^{[2]} - \alpha \frac{\partial J}{\partial W^{[2]}} \\
b^{[2]} &\leftarrow b^{[2]} - \alpha \frac{\partial J}{\partial b^{[2]}}
\end{aligned}
$$

## 6. 为什么不能全零初始化

若 $W^{[1]} = 0$，则 $Z^{[1]} = 0$，$A^{[1]} = g(0)$ 对所有神经元相同。

反向传播：
$$dZ^{[1]} = W^{[2]T} dZ^{[2]} \odot g'(0)$$

$W^{[2]T} dZ^{[2]}$ 对每一行相同，$g'(0)$ 也相同（ReLU 下为 $1$）。因此 $dZ^{[1]}$ 的每一行相同。

$$\frac{\partial J}{\partial W^{[1]}} = \frac{1}{m} dZ^{[1]} X^T$$

$dZ^{[1]}$ 每行相同 ⇒ $\frac{\partial J}{\partial W^{[1]}}$ 每行相同 ⇒ 所有神经元参数永远同步更新，等价于只有一个神经元。
