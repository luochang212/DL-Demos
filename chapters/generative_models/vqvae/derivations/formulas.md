# VQVAE 公式推导

本文档保存 VQVAE 章节的完整符号推导。网页教程负责保持可读性；这里补上向量量化、直通梯度和三段损失的中间步骤。

## 1. 符号约定

- $x$：输入图像。
- $z_e(x)$：编码器输出的连续特征。
- $\mathcal{E}=\{e_k\}_{k=1}^{K}$：码本。
- $z_q(x)$：量化后的码本向量。
- $\operatorname{sg}[\cdot]$：stop-gradient 操作。
- $D(\cdot)$：解码器。

## 2. 向量量化

对每个空间位置，选择距离编码器输出最近的码本向量：

$$
k^\*=\arg\min_j\|z_e(x)-e_j\|_2^2.
$$

量化结果为：

$$
z_q(x)=e_{k^\*}.
$$

代码中先把编码器输出和码本向量 broadcast 到同一形状，再沿通道维度求平方距离：

```python
distance = torch.sum((embedding_broadcast - ze_broadcast) ** 2, 2)
nearest_neighbor = torch.argmin(distance, 1)
```

## 3. 直通梯度

最近邻选择不可微。VQVAE 使用直通梯度估计：

$$
\text{decoder\_input}=z_e+\operatorname{sg}[z_q-z_e].
$$

前向传播时：

$$
z_e+(z_q-z_e)=z_q.
$$

反向传播时，$\operatorname{sg}[z_q-z_e]$ 的梯度为 $0$，所以梯度从解码器直接流向 $z_e$。

## 4. 三段损失

VQVAE 损失为：

$$
\mathcal{L}
=
\|x-D(z_q)\|_2^2
+
\| \operatorname{sg}[z_e]-z_q \|_2^2
+
\beta\|z_e-\operatorname{sg}[z_q]\|_2^2.
$$

三项含义：

- 重建损失：训练编码器和解码器重建图像。
- 嵌入损失：只更新码本，使码本向量靠近编码器输出。
- 承诺损失：只更新编码器，使编码器输出靠近选中的码本向量。

对应代码：

```python
l_reconstruct = mse_loss(x, x_hat)
l_embedding = mse_loss(ze.detach(), zq)
l_commitment = mse_loss(ze, zq.detach())
loss = l_reconstruct + l_w_embedding * l_embedding + l_w_commitment * l_commitment
```

## 5. 两阶段生成

训练好 VQ-VAE 后，编码器会把图像压缩成离散索引图。第二阶段训练 PixelCNN 建模这些离散索引：

$$
p(k)=\prod_i p(k_i\mid k_{<i}).
$$

生成时先用 PixelCNN 采样离散索引，再用 VQ-VAE 解码器生成图像。

## 6. Lean 已验证的恒等式

同目录的 `vqvae.lean` 验证：

- 直通梯度前向值的代数恒等式 $z_e+(z_q-z_e)=z_q$。
- 三段损失求和的简单重排不改变总损失。

Lean 文件不形式化 stop-gradient 的自动微分语义；这里只检查和公式书写直接相关的代数部分。
