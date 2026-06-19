# Attention 公式推导

本文档保存注意力机制章节的完整符号推导。网页教程负责保持可读性；这里补上从对齐分数到代码损失的中间步骤。

## 1. 符号约定

- $x^{(t')}$：输入序列第 $t'$ 个元素（日期字符串中的字符），$t' = 1, ..., T_x$
- $y^{(t)}$：输出序列第 $t$ 个元素（标准格式日期字符），$t = 1, ..., T_y$
- $a^{(t')} \in \mathbb{R}^{2d_h}$：双向 Encoder 在第 $t'$ 个位置的隐变量（拼接前后向）
- $s^{(t)} \in \mathbb{R}^{d_h}$：Decoder 在第 $t$ 步的隐状态
- $e_{t,t'} \in \mathbb{R}$：第 $t$ 个输出对第 $t'$ 个输入的**对齐分数**（alignment score）
- $\alpha_{t,t'} \in [0, 1]$：归一化后的**注意力权重**
- $c^{(t)} \in \mathbb{R}^{2d_h}$：第 $t$ 步的**上下文向量**（context vector）

代码中的变量名：
- `EMBEDDING_LENGTH`：词表大小 128（ASCII 字符集）
- `OUTPUT_LENGTH`：固定输出长度 10（`yyyy-mm-dd`）
- `encoder_dim`：单向 LSTM 隐单元数（默认 32），双向输出为 `2 × 32 = 64`
- `decoder_dim`：Decoder LSTM 隐单元数（默认 32）
- `a`：Encoder 隐变量
- `prev_s`：Decoder 上一轮状态 $s^{(t-1)}$
- `alpha`：注意力权重 $\alpha_{t,t'}$
- `c`：上下文向量 $c^{(t)}$

## 2. Encoder 的双向表示

用双向 LSTM 对输入序列编码：

$$
\begin{aligned}
\overrightarrow{a}^{(t')} &= \text{LSTM}_{\text{fwd}}(x^{(t')}, \overrightarrow{a}^{(t'-1)}) \\
\overleftarrow{a}^{(t')} &= \text{LSTM}_{\text{bwd}}(x^{(t')}, \overleftarrow{a}^{(t'+1)}) \\
a^{(t')} &= [\overrightarrow{a}^{(t')}; \overleftarrow{a}^{(t')}] \in \mathbb{R}^{2d_h}
\end{aligned}
$$

拼接后 $a^{(t')}$ 同时包含上、下文信息。

## 3. 对齐分数的计算

对齐分数由一个小型全连接网络计算，输入为 Decoder 上一状态 $s^{(t-1)}$ 和 Encoder 位置 $t'$ 的隐变量 $a^{(t')}$：

$$
e_{t,t'} = W_a [s^{(t-1)}; a^{(t')}] + b_a \in \mathbb{R}
$$

其中 $W_a \in \mathbb{R}^{1 \times (d_h + 2d_h)}$，$b_a \in \mathbb{R}$。

代码对应：

```python
attention_input = torch.cat((repeat_s, a), 2)  # [batch, n_seq, 2*enc_dim + dec_dim]
alpha = F.softmax(self.attention_linear(attention_input.reshape(...)), -1)
```

其中 `self.attention_linear = nn.Linear(2 * encoder_dim + decoder_dim, 1)`。

**并行化技巧**：将 `batch × n_sequence` 组拼接向量合并为一批，一次性通过 `attention_linear`，避免显式双重循环。

## 4. Softmax 归一化

$$
\alpha_{t,t'} = \frac{\exp(e_{t,t'})}{\sum_{k=1}^{T_x} \exp(e_{t,k})}
$$

性质：对任意 $t$，$\sum_{t'} \alpha_{t,t'} = 1$。$\alpha_{t,t'}$ 可解释为"第 $t$ 步输出应关注输入位置 $t'$ 的概率"。

Padding 位置的处理：将 padding 位置的对齐分数设为 $-\infty$，则 $\exp(-\infty) = 0$，softmax 权重自然为 0。

## 5. 上下文向量的加权平均

$$
c^{(t)} = \sum_{t'=1}^{T_x} \alpha_{t,t'} \cdot a^{(t')} \in \mathbb{R}^{2d_h}
$$

这是以注意力权重为系数的加权平均。代码：

```python
c = torch.sum(a * alpha.reshape(batch, n_sequence, 1), 1)
```

其中 `alpha` 形状为 `[batch, n_sequence]`，需要 `unsqueeze` 为 `[batch, n_sequence, 1]` 以实现广播乘法。

## 6. Decoder 的循环生成

Decoder 每步接收拼接后的输入：

$$
\text{decoder\_input}^{(t)} = [y^{(t-1)}; c^{(t)}] \in \mathbb{R}^{d_{\text{vocab}} + 2d_h}
$$

$$
s^{(t)} = \text{LSTM}(s^{(t-1)}, \text{decoder\_input}^{(t)})
$$

$$
\hat{y}^{(t)} = W_o s^{(t)} + b_o \in \mathbb{R}^{d_{\text{vocab}}}
$$

其中 $W_o \in \mathbb{R}^{d_{\text{vocab}} \times d_h}$ 是输出投影（`output_linear`）。

最终 $\hat{y}^{(t)}$ 经 softmax（由 CrossEntropyLoss 内部完成）给出下一个字符的概率分布。

**关键**：本实现中 Decoder 不接收上一轮的预测字符 $\hat{y}^{(t-1)}$，只接收注意力上下文 $c^{(t)}$。这是原论文编程作业的简化设计。标准的注意力模型将两者拼接输入。

## 7. 训练目标：交叉熵

对每个输出位置 $t$：

$$
\mathcal{L}^{(t)} = -\log P_{\text{model}}(y^{(t)} \mid x, y^{(1)}, ..., y^{(t-1)})
$$

总损失：

$$
\mathcal{L} = \sum_{t=1}^{T_y} \mathcal{L}^{(t)}
$$

代码实现：

```python
hat_y = torch.reshape(hat_y, (n * Tx, -1))
label_y = torch.reshape(y, (n * Tx,))
loss = citerion(hat_y, label_y)
```

## 8. Lean 已验证的恒等式

Lean 文件 `derivations/attention.lean` 验证了以下恒等式：

1. **Softmax 归一化**：两个元素的 softmax 权重之和为 1
2. **加权平均无偏性**：当所有权重相等时，加权平均等于普通平均
3. **维度恒等式**：Decoder 输入维度 = 词表大小 + 双向 Encoder 隐变量维度
