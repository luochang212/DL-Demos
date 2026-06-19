# Logistic Regression — 完整公式推导

## 1. Sigmoid 函数

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### Sigmoid 的导数

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**推导**：

$$
\begin{aligned}
\sigma'(z) &= \frac{d}{dz} \left(1 + e^{-z}\right)^{-1} \\
&= -\left(1 + e^{-z}\right)^{-2} \cdot (-e^{-z}) \\
&= \frac{e^{-z}}{(1 + e^{-z})^2} \\
&= \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} \\
&= \sigma(z) \cdot \frac{e^{-z}}{1 + e^{-z}} \\
&= \sigma(z) \cdot \left(1 - \frac{1}{1 + e^{-z}}\right) \\
&= \sigma(z)(1 - \sigma(z))
\end{aligned}
$$

## 2. 二元交叉熵损失

### 从 Bernoulli 极大似然出发

设 $y \in \{0, 1\}$，$\hat{y} = P(y=1 \mid x)$。Bernoulli 分布的 PMF：

$$P(y \mid x) = \hat{y}^y (1 - \hat{y})^{1-y}$$

对 $m$ 个独立样本，似然函数为：

$$\mathcal{L} = \prod_{i=1}^m \hat{y}_i^{y_i} (1 - \hat{y}_i)^{1-y_i}$$

取负对数（负对数似然）：

$$L = -\frac{1}{m} \log \mathcal{L} = -\frac{1}{m} \sum_{i=1}^m \left[y_i \log \hat{y}_i + (1-y_i) \log (1-\hat{y}_i)\right]$$

## 3. 梯度推导

### 损失对 $\hat{y}$ 的偏导

$$\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$

### Sigmoid 导数（见上文）

$$\frac{d\hat{y}}{dz} = \hat{y}(1-\hat{y})$$

### 链式法则：损失对 logit $z$

$$
\begin{aligned}
\frac{\partial L}{\partial z} &= \frac{\partial L}{\partial \hat{y}} \cdot \frac{d\hat{y}}{dz} \\
&= \left(-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}\right) \cdot \hat{y}(1-\hat{y}) \\
&= -\frac{y}{\hat{y}} \cdot \hat{y}(1-\hat{y}) + \frac{1-y}{1-\hat{y}} \cdot \hat{y}(1-\hat{y}) \\
&= -y(1-\hat{y}) + (1-y)\hat{y} \\
&= -y + y\hat{y} + \hat{y} - y\hat{y} \\
&= \hat{y} - y
\end{aligned}
$$

**关键结论**：Sigmoid 导数 $\hat{y}(1-\hat{y})$ 精确消去了 BCE 分母中的 $\hat{y}$ 和 $1-\hat{y}$，留下极其简洁的形式 $\hat{y} - y$。这是 Sigmoid + 交叉熵组合成为分类问题标准配置的根本原因。

### 损失对参数的偏导

$$z = w^T x + b$$

$$\frac{\partial z}{\partial w} = x, \qquad \frac{\partial z}{\partial b} = 1$$

对 $m$ 个样本求平均：

$$\frac{\partial L}{\partial w} = \frac{1}{m} X(\hat{Y} - Y)^T$$

$$\frac{\partial L}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i)$$

其中 $X \in \mathbb{R}^{d \times m}$（列向量为样本），$\hat{Y}, Y \in \mathbb{R}^{1 \times m}$（行向量）。

## 4. 梯度下降更新

$$w \leftarrow w - \alpha \frac{\partial L}{\partial w}$$

$$b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$

其中 $\alpha$ 为学习率。
