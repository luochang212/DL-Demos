# Transformer 公式推导

本文档保存 Transformer 章节的完整符号推导。

## 1. 符号约定

- $d_{\text{model}}$：模型隐藏维度（默认 512）
- $h$：注意力头数（默认 8）
- $d_k = d_v = d_{\text{model}} / h$：每个头的维度（默认 64）
- $d_{ff}$：前馈网络隐藏维度（默认 2048）

代码中的变量名：
- `d_model`：$d_{\text{model}}$
- `heads`：$h$
- `d_k`：每个头的 Key/Query 维度
- `d_ff` 或 `d_hidden`：前馈网络维度

## 2. Scaled Dot-Product Attention

单头 Attention 的完整计算：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**为什么除以 $\sqrt{d_k}$？** 假设 $Q$ 和 $K$ 的各分量独立，均值为 0，方差为 1。则点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的均值为 0，方差为 $d_k$。除以 $\sqrt{d_k}$ 将方差归一化为 1，防止 softmax 进入饱和区（梯度近似为 0）。

## 3. Multi-Head Attention

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
$$

其中：
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$

**代码优化**：将 $h$ 个头的线性投影合并为一次矩阵乘法。所有头的 $W^Q$ 拼接为 $W^Q_{\text{all}} \in \mathbb{R}^{d_{\text{model}} \times h d_k}$，一次性投影后再 `reshape` 拆分：

```python
Q = self.W_q(x).reshape(batch, seq_len, heads, d_k).transpose(1, 2)
# Q: [batch, heads, seq_len, d_k]
```

## 4. 位置编码

第 $pos$ 个位置、第 $i$ 个维度的编码：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

其中 $i = 0, 1, ..., d_{\text{model}}/2 - 1$。

**性质**：对于任意偏移 $k$，$PE_{pos+k}$ 可表示为 $PE_{pos}$ 的线性变换：

$$
\begin{bmatrix} \sin(pos+k) \\ \cos(pos+k) \end{bmatrix} =
\begin{bmatrix} \cos k & \sin k \\ -\sin k & \cos k \end{bmatrix}
\begin{bmatrix} \sin(pos) \\ \cos(pos) \end{bmatrix}
$$

这使得模型可以通过学习线性变换来关注相对位置。

## 5. 前馈网络（Position-wise Feed-Forward）

每个位置独立应用同一个两层全连接网络：

$$
\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2
$$

其中 $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$，$W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$，$d_{ff} = 2048$。

## 6. Layer Normalization

Transformer 使用 Post-LN（先 Attention/FFN，后 Add & Norm）：

$$
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

LayerNorm 在最后一维 $d_{\text{model}}$ 上归一化，对每个样本独立计算均值和方差：

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

## 7. Encoder 堆叠

每个 Encoder Layer 包含两个子层：

1. Multi-Head Self-Attention → Add & Norm
2. Feed-Forward → Add & Norm

Encoder 堆叠 $N$ 层（默认 $N = 6$），输入通过所有层后输出给 Decoder 的 Cross-Attention。

## 8. Decoder 堆叠

每个 Decoder Layer 包含三个子层：

1. Masked Multi-Head Self-Attention → Add & Norm
2. Cross-Attention（Q 来自 Decoder，K/V 来自 Encoder）→ Add & Norm
3. Feed-Forward → Add & Norm

Decoder Self-Attention 使用 **Causal Mask**（下三角 mask），确保位置 $t$ 只能关注 $1, ..., t$。

## 9. 训练目标：交叉熵

Decoder 最终输出 logits $\in \mathbb{R}^{B \times T \times |V|}$（$|V|$ 为中文词表大小）：

$$
\mathcal{L} = -\frac{1}{B \cdot T} \sum_{b=1}^{B} \sum_{t=1}^{T} \log P_{\text{model}}(y_{b,t} \mid x_b, y_{b,1}, ..., y_{b,t-1})
$$

代码中使用 `CrossEntropyLoss(ignore_index=PAD_ID)` 忽略 padding 位置的损失。

## 10. Lean 已验证的恒等式

Lean 文件 `derivations/transformer.lean` 验证了：

1. **缩放因子恒等式**：$1/\sqrt{64} = 1/8$（当 $d_k = 64$）
2. **多头维度恒等式**：$h \cdot d_k = d_{\text{model}}$（如 $8 \times 64 = 512$）
3. **位置编码周期性**：$\sin^2(x) + \cos^2(x) = 1$
