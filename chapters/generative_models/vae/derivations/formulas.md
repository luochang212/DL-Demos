# VAE 公式推导

本文档保存 VAE 章节的完整符号推导。网页教程负责保持可读性；这里补上从概率目标到代码损失的中间步骤，方便想继续深挖的读者阅读。

## 1. 符号约定

- $x$：观测图像。
- $z$：隐变量。
- $p(z)=\mathcal{N}(0,I)$：隐变量先验。
- $p_\theta(x\mid z)$：解码器定义的生成分布。
- $q_\phi(z\mid x)$：编码器给出的后验近似。
- $q_\phi(z\mid x)=\mathcal{N}(\mu_\phi(x), \operatorname{diag}(\sigma_\phi^2(x)))$。
- `mean`：代码中的 $\mu$。
- `logvar`：代码中的 $\log\sigma^2$。

## 2. ELBO 推导

目标是最大化观测数据的边缘对数似然：

$$
\log p_\theta(x)=\log\int p_\theta(x,z)\,dz.
$$

插入任意一个支撑集覆盖真实后验的分布 $q_\phi(z\mid x)$：

$$
\log p_\theta(x)
= \log\int q_\phi(z\mid x)\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz.
$$

这可以写成 $q_\phi(z\mid x)$ 下的期望：

$$
\log p_\theta(x)
= \log\mathbb{E}_{q_\phi(z\mid x)}
\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right].
$$

由 Jensen 不等式：

$$
\log p_\theta(x)
\ge
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right].
$$

展开联合分布 $p_\theta(x,z)=p_\theta(x\mid z)p(z)$：

$$
\mathcal{L}_{\mathrm{ELBO}}
=
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-
D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z)).
$$

因此，最大化 ELBO 等价于同时做两件事：

- 提高重建概率：给定 $z$ 后，解码器应当能生成接近 $x$ 的图像。
- 降低 KL 散度：编码器输出的分布应当接近先验 $p(z)$。

教程代码实现的是负 ELBO 形式的损失：

$$
\mathrm{loss}
=
\mathrm{reconstruction\ loss}
+
\lambda D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z)).
$$

## 3. 为什么重建误差可以用 MSE

如果假设解码器似然是固定方差的各向同性高斯分布：

$$
p_\theta(x\mid z)=\mathcal{N}(\hat{x}_\theta(z), \sigma_x^2 I),
$$

那么负对数似然为：

$$
-\log p_\theta(x\mid z)
=
\frac{1}{2\sigma_x^2}\|x-\hat{x}_\theta(z)\|_2^2 + C,
$$

其中 $C$ 与模型参数无关。当 $\sigma_x^2$ 固定时，最小化负对数似然就等价于在常数尺度下最小化 MSE。

这对应代码：

```python
recons_loss = F.mse_loss(y_hat, y)
```

## 4. 对角高斯的 KL 项

先看一个隐变量维度：

$$
q(z)=\mathcal{N}(\mu,\sigma^2),\qquad p(z)=\mathcal{N}(0,1).
$$

KL 散度有解析式：

$$
D_{\mathrm{KL}}(q\|p)
=
\frac{1}{2}\left(\mu^2+\sigma^2-\log\sigma^2-1\right).
$$

对角高斯各维度独立，所以对所有维度求和：

$$
D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))
=
\frac{1}{2}
\sum_j
\left(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1\right).
$$

实现里保存的是 $\log\sigma^2$，也就是 `logvar`，因此 $\sigma^2=\exp(\texttt{logvar})$：

$$
D_{\mathrm{KL}}
=
-\frac{1}{2}
\sum_j
\left(1+\texttt{logvar}_j-\texttt{mean}_j^2-\exp(\texttt{logvar}_j)\right).
$$

这正对应代码：

```python
kl_loss = torch.mean(
    -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0
)
```

## 5. 重参数化技巧

直接从 $q_\phi(z\mid x)$ 采样会让梯度路径经过随机采样操作。VAE 把随机性移到一个独立噪声变量上：

$$
\epsilon\sim\mathcal{N}(0,I),\qquad
z=\mu+\sigma\odot\epsilon.
$$

因为 $\epsilon$ 与模型参数无关，梯度可以通过 $\mu$ 和 $\sigma$ 正常反向传播。

代码中：

```python
std = torch.exp(logvar / 2)
z = eps * std + mean
```

## 6. Lean 已验证的恒等式

同目录的 `vae.lean` 验证了 KL 项代码形式使用的代数恒等式：

$$
-(1+\ell-m-v)=m+v-\ell-1.
$$

其中 $m$ 表示 $\mu^2$，$v$ 表示 $\sigma^2$，$\ell$ 表示 $\log\sigma^2$。这个检查用于防止把标准 KL 公式翻译成代码形式时写错符号或项顺序。

Lean 文件不试图形式化 Jensen 不等式或高斯测度论。这些是 VAE 论文和概率论中的标准结论；本地 Lean 校验只覆盖和代码直接相关的代数部分。
