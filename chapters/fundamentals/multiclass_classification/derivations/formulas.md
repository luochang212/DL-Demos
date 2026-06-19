# 多分类 — Softmax 与交叉熵完整推导

## 1. Softmax 函数

给定 logits $z \in \mathbb{R}^C$（$C$ 个类别的原始分数）：

$$a_k = \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^C e^{z_j}}, \quad k = 1, \ldots, C$$

性质：$a_k \in (0, 1)$，$\sum_{k=1}^C a_k = 1$。

### 退化到 Sigmoid（$C=2$）

设 $z = [z_1, z_2]$：

$$a_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2}} = \frac{1}{1 + e^{-(z_1 - z_2)}} = \sigma(z_1 - z_2)$$

令 $z = z_1 - z_2$，则 $a_1 = \sigma(z)$——与二分类 sigmoid 一致。

## 2. 多分类交叉熵损失

### One-hot 编码

将类别标签 $y \in \{0, 1, \ldots, C-1\}$ 转为向量 $\mathbf{y} \in \{0, 1\}^C$，只有正确类别位置为 $1$。

### 损失函数

$$L = -\sum_{k=1}^C y_k \log a_k = -\log a_{y_{\text{true}}}$$

对 $m$ 个样本求平均：

$$J = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^C y_k^{(i)} \log a_k^{(i)}$$

## 3. 梯度推导

### Softmax 的 Jacobian

$$\frac{\partial a_k}{\partial z_j} = a_k (\delta_{kj} - a_j)$$

推导：

$\frac{\partial a_k}{\partial z_j} = \frac{\partial}{\partial z_j} \frac{e^{z_k}}{\sum_l e^{z_l}}$

当 $k = j$：
$$\frac{\partial a_k}{\partial z_k} = \frac{e^{z_k} \sum_l e^{z_l} - e^{z_k} \cdot e^{z_k}}{(\sum_l e^{z_l})^2} = a_k - a_k^2 = a_k(1 - a_k)$$

当 $k \neq j$：
$$\frac{\partial a_k}{\partial z_j} = \frac{0 - e^{z_k} \cdot e^{z_j}}{(\sum_l e^{z_l})^2} = -a_k a_j$$

统一形式：$\frac{\partial a_k}{\partial z_j} = a_k(\delta_{kj} - a_j)$。

### 链式法则

$$\frac{\partial L}{\partial z_j} = \sum_{k=1}^C \frac{\partial L}{\partial a_k} \frac{\partial a_k}{\partial z_j}$$

其中 $\frac{\partial L}{\partial a_k} = -\frac{y_k}{a_k}$。

代入：

$$\frac{\partial L}{\partial z_j} = \sum_{k=1}^C -\frac{y_k}{a_k} \cdot a_k(\delta_{kj} - a_j) = -\sum_{k=1}^C y_k(\delta_{kj} - a_j)$$

$$= -\left(y_j - a_j\sum_{k=1}^C y_k\right) = -(y_j - a_j) = a_j - y_j$$

最后一步用了 $\sum_k y_k = 1$（one-hot 性质）。

### 结论

$$\frac{\partial L}{\partial z} = a - y$$

这与 Sigmoid + BCE 的梯度形式完全一致——「预测减标签」。这是交叉熵损失族的普遍性质，与具体的激活函数（Sigmoid / Softmax）无关。

## 4. 实现要点

### Logits vs Probabilities

PyTorch `CrossEntropyLoss` 内部等价于 `LogSoftmax + NLLLoss`。因此：

```python
# ✅ 正确：模型返回 logits，loss 内部处理 softmax
logits = model(x)
loss = CrossEntropyLoss()(logits, labels)

# ❌ 错误：先 softmax 再传给 CrossEntropyLoss
probs = F.softmax(logits, dim=1)
loss = CrossEntropyLoss()(probs, labels)  # 会再做一次 log-softmax
```
