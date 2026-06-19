# Fourier Feature 公式推导

本文档保存傅里叶特征章节的完整符号推导。网页教程负责保持可读性；这里补上从频谱偏置到核函数展开的中间步骤。

> 工程类章节（engineering）走非生成式轻量变体，公式推导不绑定 Lean4 校验。编译命令仅 `lake build`，不新增 `.lean` 文件。

## 1. 符号约定

- $x, x' \in \mathbb{R}^d$：输入空间中的两个坐标向量。对于图像拟合任务，$d = 2$（即 $(u, v)$ 像素坐标）
- $B \in \mathbb{R}^{m \times d}$：随机傅里叶基矩阵，$B_{ij} \sim \mathcal{N}(0, \sigma^2)$ 采样后冻结
- $\sigma$：傅里叶特征的 scale 参数，控制频率范围
- $f_\theta(x)$：参数为 $\theta$ 的 MLP；$f_\theta : \mathbb{R}^d \to \mathbb{R}^c$
- $K(x, x')$：Neural Tangent Kernel（NTK），描述无限宽网络的训练动态
- $\gamma(x)$：Fourier Feature 映射 $\gamma : \mathbb{R}^d \to \mathbb{R}^{2m}$

代码中的变量名：

- `in_c`：输入维度 $d$（默认 2）
- `out_c`：映射后维度 $2m$（默认 256，即 $m = 128$）
- `scale`：$\sigma$（默认 10）
- `_fourier_basis`：随机矩阵 $B$，形状 `(in_c, out_c // 2)`
- `N, C, H, W`：batch、通道、图像高、图像宽

## 2. 问题设定：MLP 拟合坐标到颜色

任务描述：给定一张 $H \times W$ 的 RGB 图像 $I$，用一个 MLP $f_\theta$ 对每个像素坐标映射其颜色值。

- **输入**：归一化坐标 $x = (u, v) \in [0, 1]^2$
- **目标**：$y = I(u, v) \in [0, 1]^3$
- **损失**：L1 损失 $\mathcal{L} = \mathbb{E}_{u,v} \|f_\theta(u, v) - I(u, v)\|_1$

直觉上，足够宽的 MLP 能拟合任意连续函数，但实验发现：**标准 MLP 能快速学到图片的低频轮廓，却很难捕捉高频细节（纹理、边缘）**。

## 3. 频谱偏置与 NTK 理论

### 3.1 Neural Tangent Kernel 的定义

在无限宽极限下，用梯度下降训练 MLP 的动态等价于用如下核函数进行核回归：

$$
K(x, x') = \left\langle \nabla_\theta f_\theta(x),\ \nabla_\theta f_\theta(x') \right\rangle
$$

其中 $\nabla_\theta f_\theta(x)$ 是 MLP 输出对全体参数 $\theta$ 的梯度在 $x$ 处的值。初始化的随机性确定后，$K$ 在训练中近似不变。

### 3.2 NTK 的特征分解

考虑 NTK 在输入分布上的特征分解：

$$
K(x, x') = \sum_{k=0}^{\infty} \lambda_k\ \phi_k(x)\ \phi_k(x')
$$

其中 $\{\phi_k\}$ 是 $L^2$ 空间的标准正交基（按频率 $k$ 排序），$\lambda_k \ge 0$ 是对应特征值。

梯度下降在第 $k$ 个频率分量的收敛速度为：

$$
\|\hat{f}_k^{(t)} - f_k^*\| \propto e^{-\eta \lambda_k t}
$$

因此**特征值 $\lambda_k$ 越大的分量学得越快**。

### 3.3 频谱偏置的来源

对于标准的 ReLU MLP，经验与理论分析（Jacot et al. 2018，Tancik et al. 2020）表明：

$$
\lambda_k \sim O(k^{-\alpha}),\quad \alpha > 1
$$

NTK 的特征值随频率 $k$ **指数级别衰减**。这意味着：

- 低频分量（$k$ 小）→ $\lambda_k$ 大 → 快速收敛
- 高频分量（$k$ 大）→ $\lambda_k$ 极小 → 极慢收敛

这就是**频谱偏置**（Spectral Bias）的 NTK 解释。

## 4. 随机 Fourier 特征映射

### 4.1 构造

Fourier Feature 映射将一个 $d$ 维输入向量 $v$ 变换为 $2m$ 维特征：

$$
\gamma(v) = \begin{bmatrix}
\cos(2\pi B_1 v)\\
\vdots\\
\cos(2\pi B_m v)\\
\sin(2\pi B_1 v)\\
\vdots\\
\sin(2\pi B_m v)
\end{bmatrix}
= \begin{bmatrix}
\cos(2\pi B v)\\
\sin(2\pi B v)
\end{bmatrix}
\in \mathbb{R}^{2m}
$$

其中 $B_i \in \mathbb{R}^{1 \times d}$ 是 $B$ 的第 $i$ 行，$B \in \mathbb{R}^{m \times d}$。

代码实现：

```python
class FourierFeature(nn.Module):
    def __init__(self, in_c: int, out_c: int, scale: float):
        super().__init__()
        # B ∈ R^{in_c × (out_c//2)},  B_ij ~ N(0, σ²)
        fourier_basis = torch.randn(in_c, out_c // 2) * scale
        self.register_buffer('_fourier_basis', fourier_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        x = rearrange(x, 'n c h w -> (n h w) c')      # (N·H·W, 2)
        x = x @ self._fourier_basis                     # (N·H·W, 128)
        x = rearrange(x, '(n h w) c -> n c h w', h=H, w=W)
        x = 2 * torch.pi * x
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=1)  # (N, 256, H, W)
        return x
```

关键维度关系：

$$
2 \cdot \left\lfloor \frac{\text{out\_c}}{2} \right\rfloor = \text{out\_c}
$$

当 `out_c` 为偶数时精确成立。代码中取 `out_c // 2` 作 $B$ 的列数，Cat 后维度翻倍回到 `out_c`。

### 4.2 映射后的核函数——积化和差展开

将 $\gamma(x)$ 送入 MLP 后，新模型的 NTK 变为：

$$
K_\gamma(x, x') = K_{\text{NTK}}(\gamma(x),\ \gamma(x'))
$$

对于深层 ReLU MLP，$K_{\text{NTK}}$ 是输入内积的逐点非线性函数。在 $\gamma$ 映射下，内积为：

$$
\begin{aligned}
\langle \gamma(x),\ \gamma(x') \rangle
&= \sum_{i=1}^{m} \Big[ \cos(2\pi B_i x) \cos(2\pi B_i x') + \sin(2\pi B_i x) \sin(2\pi B_i x') \Big] \\[4pt]
&= \sum_{i=1}^{m} \cos\!\big(2\pi B_i(x - x')\big)
\end{aligned}
$$

**关键推导步骤**（积化和差）：

$$
\cos a \cos b + \sin a \sin b = \cos(a - b)
$$

令 $a = 2\pi B_i x$，$b = 2\pi B_i x'$，代入得：

$$
\cos(2\pi B_i x) \cos(2\pi B_i x') + \sin(2\pi B_i x) \sin(2\pi B_i x') = \cos\!\big(2\pi B_i (x - x')\big)
$$

逐 $i$ 求和即得上述结论。

### 4.3 核心结论：核变为平移不变函数

映射前：$K(x, x')$ 依赖 $x$ 和 $x'$ 的绝对值，NTK 类似"低通滤波器"。

映射后：

$$
K_\gamma(x, x') = h\!\left(\sum_{i=1}^{m} \cos\!\big(2\pi B_i (x - x')\big)\right)
$$

其中 $h$ 是固定非线性函数（由 MLP 深度和激活函数决定，在无限宽极限下可显式表达）。

**核函数只依赖差值 $x - x'$**。这意味着：

1. **平移不变性**：$K_\gamma(x + \Delta, x' + \Delta) = K_\gamma(x, x')$
2. **频率可控**：$B$ 的分布决定 $K_\gamma$ 对哪些频率的输入差敏感

### 4.4 Random Features 与 Bochner 定理

这一构造源自 Rahimi & Recht（2007）的随机特征方法。

**Bochner 定理**：任意正定平移不变核 $k(x - x')$ 存在唯一的有限正测度 $p(w) \, dw$，使得：

$$
k(x - x') = \int_{\mathbb{R}^d} e^{i\omega^\top (x - x')} \, p(\omega) \, d\omega
$$

取实部（核是实值函数）：

$$
k(x - x') = \int_{\mathbb{R}^d} \cos\!\big(\omega^\top (x - x')\big) \, p(\omega) \, d\omega
$$

用 $m$ 个 Monte Carlo 样本 $\{B_i\}_{i=1}^m \sim p(\omega)$ 近似：

$$
k(x - x') \approx \frac{1}{m} \sum_{i=1}^{m} \cos\!\big(B_i (x - x')\big)
$$

当 $p(\omega) = \mathcal{N}(0, \sigma^2 I)$ 时，对应的平移不变核是**高斯核**：

$$
k(x - x') = \exp\!\left(-\frac{\sigma^2 \|x - x'\|^2}{2}\right)
$$

Tancik et al.（2020）将这一框架应用于 NTK：通过在 MLP 前插入 $\gamma$，网络的 NTK 变为近似平移不变量，且其频率响应由 $p(\omega)$ 塑形。

## 5. scale 参数 $\sigma$ 的控制作用

$\sigma$ 决定了 $B$ 的元素尺度，从而控制 $\gamma$ 对哪些频率的输入差敏感。

### 5.1 理论分析

令 $\omega_i = B_i^\top \in \mathbb{R}^d$，则：

$$
K_\gamma(x, x') \approx h\!\left( \frac{1}{m} \sum_{i=1}^{m} \cos(\omega_i^\top (x - x')) \right)
$$

每个 $\omega_i$ 从 $\mathcal{N}(0, \sigma^2 I_d)$ 中独立采样，其范数 $\|\omega_i\|$ 的分布决定映射对输入差的敏感频率。

- $\|\omega\|$ 在半径 $\approx \sigma\sqrt{d}$ 附近概率密度最大
- 对应的空间频率为 $f_{\text{peak}} \approx \sigma \sqrt{d} / (2\pi)$

### 5.2 三种 regime

| $\sigma$ | $\|\omega_i\|$ 分布 | NTK 频率响应 | 实验效果 |
|-----------|---------------------|-------------|----------|
| $\sigma$ 过小 (≈0.1) | $\omega$ 接近 0 | 近似常量，等同于不加映射 | 仍然只能学低频 |
| $\sigma$ 适中 (≈10) | $\omega$ 分布在图像纹理频率附近 | 匹配目标频率范围 | **拟合效果最好** |
| $\sigma$ 过大 (≈100) | $\omega$ 包含极高频 | 高频响应增强但可能过拟合噪声 | 纹理更锐利但可能产生伪影 |

Tancik et al. 的实验建议：$\sigma$ 应匹配目标信号的**有效空间频率**（如纹理周期、边缘梯度）。

## 6. 2D 图像输入的推广

对于图像坐标 $(u, v) \in [0, 1]^2$，$d = 2$。Fourier Feature 对每个像素独立映射：

$$
\gamma(u, v) = \begin{bmatrix}
\cos\!\big(2\pi(B_{11}u + B_{12}v)\big)\\
\vdots\\
\cos\!\big(2\pi(B_{m1}u + B_{m2}v)\big)\\
\sin\!\big(2\pi(B_{11}u + B_{12}v)\big)\\
\vdots\\
\sin\!\big(2\pi(B_{m1}u + B_{m2}v)\big)
\end{bmatrix}
$$

代码实现中使用 1×1 卷积处理 2D 网格：MLP 的 Conv2d(kernel_size=1) 对每个位置独立计算全连接，配合 Fourier Feature 的输出特征图，等价于对每个像素做带傅里叶先验的全连接回归。

### 连续视角

Fourier Feature + MLP 可视为一项**隐式神经表示**（implicit neural representation, INR）：

$$
\Phi(u, v) = \text{MLP}(\gamma(u, v))
$$

$\Phi$ 是关于连续坐标的函数，可在任意解析度上采样（不仅是训练时的 $H \times W$）。这种连续参数化的能力是 Fourier Feature 在 NeRF（Mildenhall et al. 2020）等领域获得成功的基础。

## 7. 平移不变性实验（频率空间）

Fourier Feature 的一个重要性质：**空间域的平移对应于频率空间中 phase（相位）的偏移**。

设 $B$ 的每行 $B_i$，映射后向量在频率空间中的 phase 为 $\theta_i(x) = 2\pi B_i x$。输入平移 $x \mapsto x + \Delta$ 导致：

$$
\theta_i(x + \Delta) = 2\pi B_i (x + \Delta) = \theta_i(x) + 2\pi B_i \Delta
$$

即相位偏移 $2\pi B_i \Delta$。利用这一性质，可以在频率空间中实现仿射变换（见 `image_mlp.ipynb` 中的 `aff_transform` 函数）。

## 8. 参考资料

1. **Tancik, M., Srinivasan, P. P., Mildenhall, B., et al.** (2020). [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739). *NeurIPS*.
   - 提出用随机 Fourier Feature 映射突破 MLP 的频谱偏置，给出 NTK 频率分析。

2. **Rahimi, A. & Recht, B.** (2007). [Random Features for Large-Scale Kernel Machines](https://proceedings.neurips.cc/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf). *NeurIPS*.
   - 随机特征方法（Random Kitchen Sinks）的起源：用随机投影+余弦近似任意平移不变核。

3. **Jacot, A., Gabriel, F., & Hongler, C.** (2018). [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572). *NeurIPS*.
   - NTK 理论的开创性工作：确立无限宽网络的训练动态由核回归描述。

4. **Mildenhall, B., Srinivasan, P. P., Tancik, M., et al.** (2020). [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934). *ECCV*.
   - 将 Fourier Feature（Positional Encoding）应用于 3D 场景的隐式神经表示，实现高保真新视角合成。
