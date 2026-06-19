# RNN 公式推导

本文档保存 basic_rnn 章节的完整符号推导。网页教程负责保持可读性；这里补上从 RNN 前向公式到代码损失的中间步骤，方便想继续深挖的读者阅读。

## 1. 符号约定

- $x^{(t)} \in \mathbb{R}^{d_x}$：时间步 $t$ 的输入向量。对于字符级语言模型，$x^{(t)}$ 是当前字符的 one-hot 编码，$d_x = 27$（26 个字母 + 空格）。
- $a^{(t)} \in \mathbb{R}^{d_h}$：时间步 $t$ 的隐藏状态（hidden state），$d_h$ 为隐藏单元数。
- $\hat{y}^{(t)} \in \mathbb{R}^{d_x}$：时间步 $t$ 的输出（下一个字符的概率 logits）。
- $W_{ax} \in \mathbb{R}^{d_h \times d_x}$、$W_{aa} \in \mathbb{R}^{d_h \times d_h}$：输入到隐藏、隐藏到隐藏的权重矩阵。
- $W_{ya} \in \mathbb{R}^{d_x \times d_h}$：隐藏到输出的权重矩阵。
- $b_a \in \mathbb{R}^{d_h}$、$b_y \in \mathbb{R}^{d_x}$：偏置。
- $g(\cdot)$：隐藏层激活函数，通常为 $\tanh$。

代码中使用的变量名：
- `EMBEDDING_LENGTH`：$d_x = 27$
- `hidden_units`：$d_h$（RNN1 默认 32，RNN2 默认 64）
- `a`：隐藏状态 $a^{(t)}$
- `x`：当前输入 $x^{(t)}$
- `hat_y`：输出 $\hat{y}^{(t)}$

## 2. RNN 前向传播

### 2.1 标准形式

RNN 在第 $t$ 轮计算中执行：

$$
\begin{aligned}
a^{(t)} &= g(W_{ax} x^{(t)} + W_{aa} a^{(t-1)} + b_a) \\
\hat{y}^{(t)} &= W_{ya} a^{(t)} + b_y
\end{aligned}
$$

初始隐藏状态 $a^{(0)} = \mathbf{0}$（全零向量）。

### 2.2 简化拼接形式

将 $W_{ax}$ 和 $W_{aa}$ 水平拼接，$x^{(t)}$ 和 $a^{(t-1)}$ 垂直拼接，可以简化表示：

$$
\begin{aligned}
W_a &= [W_{aa} \mid W_{ax}] \in \mathbb{R}^{d_h \times (d_h + d_x)} \\
[a^{(t-1)}; x^{(t)}] &= \begin{bmatrix} a^{(t-1)} \\ \hline x^{(t)} \end{bmatrix} \in \mathbb{R}^{d_h + d_x}
\end{aligned}
$$

于是前向公式简化为一行：

$$
a^{(t)} = g(W_a [a^{(t-1)}; x^{(t)}] + b_a)
$$

这正是代码中 `self.linear_a(torch.cat((a, x), 1))` 的数学表达。`linear_a` 的输入维度为 `hidden_units + EMBEDDING_LENGTH = d_h + d_x`，输出维度为 `hidden_units = d_h`。

## 3. 语言模型的概率分解

### 3.1 链式法则

语言模型对序列 $c^{(1)}, c^{(2)}, ..., c^{(T)}$ 的概率建模：

$$
P(c^{(1)}, ..., c^{(T)}) = \prod_{t=1}^{T} P(c^{(t)} \mid c^{(1)}, ..., c^{(t-1)})
$$

RNN 的每一次输出 $\hat{y}^{(t)}$（经过 softmax）恰好拟合条件概率 $P(c^{(t)} \mid c^{(1)}, ..., c^{(t-1)})$。

### 3.2 训练目标：交叉熵

对每个时间步 $t$，模型输出 logits $\hat{y}^{(t)} \in \mathbb{R}^{d_x}$，目标为真实的下一个字符的 one-hot 标签 $y^{(t)}$。交叉熵损失为：

$$
\mathcal{L}^{(t)} = -\sum_{k=1}^{d_x} y_k^{(t)} \log \text{softmax}(\hat{y}^{(t)})_k = -\log P_{\text{model}}(c^{(t)} \mid c^{(1)}, ..., c^{(t-1)})
$$

整个序列的损失为各时间步损失之和：

$$
\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}^{(t)}
$$

在代码中，PyTorch 的 `CrossEntropyLoss` 自动包含了 softmax 和负对数似然的计算，因此模型 `forward()` 输出的是原始 logits，不包含 softmax。

### 3.3 输入与标签的错位关系

训练时，输入序列和标签序列**错开一个位置**：

| 位置 | 1 | 2 | 3 | ... | T | T+1 |
|------|---|---|---|-----|---|-----|
| 输入 $x^{(t)}$ | $\langle sos \rangle$ | $c^{(1)}$ | $c^{(2)}$ | ... | $c^{(T-1)}$ | $c^{(T)}$ |
| 标签 $y^{(t)}$ | $c^{(1)}$ | $c^{(2)}$ | $c^{(3)}$ | ... | $c^{(T)}$ | $\langle eos \rangle$ |

代码中以 `first_letter = word.new_zeros(batch, 1)` 作为 $\langle sos \rangle$（全零向量），`x = torch.cat((first_letter, word[:, 0:-1]), 1)` 完成错位。

## 4. 沿时间步的反向传播（BPTT）

RNN 的损失 $\mathcal{L}$ 对参数 $W$ 的梯度为各时间步梯度之和：

$$
\frac{\partial \mathcal{L}}{\partial W} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}^{(t)}}{\partial W}
$$

对于隐藏状态的梯度，需要沿时间步反向递推。由链式法则：

$$
\frac{\partial \mathcal{L}}{\partial a^{(t)}} = \frac{\partial \mathcal{L}^{(t)}}{\partial a^{(t)}} + \frac{\partial \mathcal{L}}{\partial a^{(t+1)}} \cdot \frac{\partial a^{(t+1)}}{\partial a^{(t)}}
$$

其中 $\frac{\partial a^{(t+1)}}{\partial a^{(t)}} = \text{diag}(g'(z^{(t+1)})) \cdot W_{aa}$（对于标准形式）。

当 $W_{aa}$ 的特征值反复相乘时，梯度可能呈指数增长（爆炸）或衰减（消失）。

## 5. 梯度裁剪

梯度爆炸的解决方案是**梯度裁剪**（gradient clipping）。设梯度向量为 $\mathbf{g} = \nabla_W \mathcal{L}$，阈值为 $\theta$：

$$
\mathbf{g}_{\text{clipped}} = \begin{cases}
\mathbf{g} & \text{if } \|\mathbf{g}\|_2 \leq \theta \\
\theta \cdot \dfrac{\mathbf{g}}{\|\mathbf{g}\|_2} & \text{if } \|\mathbf{g}\|_2 > \theta
\end{cases}
$$

裁剪后梯度方向不变，但范数不超过 $\theta$。代码中使用 `torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)`，即 $\theta = 0.5$。

**为什么阈值选 0.5？** 这是一个经验值，对于字符级语言模型这种小规模任务已经足够。实际中可在 0.1–10 之间调试。

## 6. GRU 门控方程

RNN2 使用 PyTorch 的 `nn.GRU`，其核心方程为：

$$
\begin{aligned}
r_t &= \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr}) \quad &\text{（重置门）} \\
z_t &= \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz}) \quad &\text{（更新门）} \\
n_t &= \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn})) \quad &\text{（候选隐藏状态）} \\
h_t &= (1 - z_t) \odot n_t + z_t \odot h_{t-1} \quad &\text{（最终隐藏状态）}
\end{aligned}
$$

其中 $\sigma$ 为 sigmoid 函数，$\odot$ 为逐元素乘法。更新门 $z_t$ 在 0（完全使用旧状态）和 1（完全使用新候选）之间插值，起到类似人类"选择性记忆"的作用。

## 7. Lean 已验证的恒等式

Lean 文件 `derivations/basic_rnn.lean` 验证了以下面向代码的代数恒等式：

1. **隐藏状态初始化**：当初始隐藏状态和第一轮输入均为零时，第一轮隐藏状态仅取决于偏置 $b_a$。
2. **梯度裁剪保号性**：裁剪后的梯度方向与原梯度方向一致（标量情况下的比例关系）。
3. **拼接维度恒等式**：`hidden_units + EMBEDDING_LENGTH` 等于 `linear_a` 的输入维度。
