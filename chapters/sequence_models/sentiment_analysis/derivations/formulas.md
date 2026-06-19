# 情感分析 公式推导

本文档保存情感分析章节的完整符号推导。网页教程负责保持可读性；这里补上从词嵌入到 BCE 损失的中间步骤。

## 1. 符号约定

- $x = (w_1, ..., w_T)$：输入文本序列，$T$ 为序列长度
- $y \in \{0, 1\}$：情感标签（0 = 负面，1 = 正面）
- $e(w) \in \mathbb{R}^{d_{\text{glove}}}$：GloVe 词嵌入向量，$d_{\text{glove}} = 100$
- $h_t \in \mathbb{R}^{d_h}$：GRU 在时间步 $t$ 的隐藏状态，$d_h = 64$
- $\hat{y} \in [0, 1]$：模型输出的正面概率

代码中的变量名：
- `GLOVE_DIM`：$d_{\text{glove}} = 100$
- `hidden_units`：$d_h = 64$
- `x`：输入词嵌入序列
- `lengths`：各样本的实际长度（用于 pack_padded_sequence）

## 2. 词嵌入：从 Token 到向量

GloVe 词嵌入将一个 token $w$ 映射为稠密向量：

$$
e(w) = \text{GloVe}[w] \in \mathbb{R}^{100}
$$

未登录词（OOV）映射为零向量。代码中通过 `GLOVE.get_vecs_by_tokens(sentence)` 完成映射。

## 3. GRU 门控方程

GRU 的核心计算（PyTorch `nn.GRU` 实现）：

$$
\begin{aligned}
r_t &= \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr}) \quad &\text{（重置门）} \\
z_t &= \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz}) \quad &\text{（更新门）} \\
n_t &= \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn})) \quad &\text{（候选状态）} \\
h_t &= (1 - z_t) \odot n_t + z_t \odot h_{t-1} \quad &\text{（最终状态）}
\end{aligned}
$$

## 4. 序列分类的聚合策略

对于变长序列分类任务，取**最后有效位置**的隐藏状态作为整个序列的表示：

$$
h_{\text{seq}} = h_{T_{\text{real}}} \in \mathbb{R}^{64}
$$

其中 $T_{\text{real}}$ 是样本的真实长度（不含 padding）。代码通过 `pack_padded_sequence` + `hidden[-1]` 实现：

```python
packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
_, hidden = self.rnn(packed)
output = self.linear(hidden[-1])  # 取最后层的最后一个有效状态
```

## 5. 二分类输出

$$
z = W_o h_{\text{seq}} + b_o \in \mathbb{R}, \quad W_o \in \mathbb{R}^{1 \times 64}
$$

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} \in [0, 1]
$$

其中 $\sigma$ 为 sigmoid 函数。代码：`self.sigmoid = nn.Sigmoid()`。

## 6. 训练目标：二元交叉熵

对于单个样本：

$$
\mathcal{L} = -[y \log \hat{y} + (1 - y) \log(1 - \hat{y})]
$$

代码中使用 `nn.BCELoss()`，要求模型输出已经过 sigmoid。

与 CrossEntropyLoss 的关系：二分类的 BCE 是交叉熵在 $K=2$ 时的特例。若使用 CrossEntropyLoss，则模型应输出 2 维 logits 而非 1 维 sigmoid 概率。

## 7. 变长序列的批处理

同一 batch 中序列长度不同。`pad_sequence` 将短序列填充到 batch 内最大长度，`pack_padded_sequence` 让 GRU 跳过 padding 位置：

$$
\text{x\_pad} \in \mathbb{R}^{B \times T_{\max} \times 100}
$$

填充值为 0（对应 GloVe 的零向量，与 OOV 处理一致）。

## 8. Lean 已验证的恒等式

Lean 文件 `derivations/sentiment_analysis.lean` 验证了：

1. **Sigmoid 输出范围**：当 $z = 0$ 时 $\sigma(0) = 0.5$
2. **BCE 对称性**：$\mathcal{L}(y, \hat{y}) = \mathcal{L}(1-y, 1-\hat{y})$
3. **词嵌入维度恒等式**：GRU 输入维度 = GloVe 维度
