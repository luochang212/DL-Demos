# DDPM 公式推导

本文档保存 DDPM 章节的完整符号推导。网页教程负责保持可读性；这里补上从前向加噪、后验均值/方差到噪声预测目标的中间步骤。

## 1. 符号约定

- $x_0$：干净图像。
- $x_t$：经过 $t$ 步前向加噪后的图像。
- $\epsilon$：标准高斯噪声。
- $\beta_t$：噪声方差调度。
- $\alpha_t = 1-\beta_t$。
- $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$。
- $\epsilon_\theta(x_t,t)$：神经网络预测的噪声。

## 2. 前向过程

DDPM 固定前向过程：

$$
q(x_t\mid x_{t-1})
=
\mathcal{N}
\left(
x_t;
\sqrt{\alpha_t}x_{t-1},
(1-\alpha_t)I
\right).
$$

因为 $\beta_t=1-\alpha_t$，也可以写成：

$$
q(x_t\mid x_{t-1})
=
\mathcal{N}
\left(
x_t;
\sqrt{1-\beta_t}x_{t-1},
\beta_t I
\right).
$$

采样形式为：

$$
x_t = \sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t.
$$

连续代入两步：

$$
x_t
=
\sqrt{\alpha_t\alpha_{t-1}}x_{t-2}
+
\sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_{t-1}
+
\sqrt{1-\alpha_t}\epsilon_t.
$$

独立高斯噪声的线性组合仍然是高斯噪声，两步时总方差为：

$$
1-\alpha_t\alpha_{t-1}.
$$

推广到任意 $t$：

$$
q(x_t\mid x_0)
=
\mathcal{N}
\left(
x_t;
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
\right).
$$

因此训练时可以一步采样任意时间步：

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

这就是 `sample_forward` 实现的公式。

## 3. 逆向过程与噪声预测

学习到的逆向过程定义为：

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}
\left(
x_{t-1};
\mu_\theta(x_t,t),
\sigma_t^2I
\right).
$$

由前向闭式公式：

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
$$

可以反解出 $x_0$：

$$
x_0
=
\frac{
x_t-\sqrt{1-\bar\alpha_t}\epsilon
}{
\sqrt{\bar\alpha_t}
}.
$$

采样时真实噪声 $\epsilon$ 不可见，所以用网络预测：

$$
\hat{x}_0
=
\frac{
x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)
}{
\sqrt{\bar\alpha_t}
}.
$$

这对应 `clip_x0` 分支中的代码：

```python
x_0 = (x_t - torch.sqrt(1 - self.alpha_bars[t]) * eps) / torch.sqrt(
    self.alpha_bars[t]
)
```

## 4. 后验均值和方差

给定 $x_0$ 时，真实逆向后验是高斯分布：

$$
q(x_{t-1}\mid x_t,x_0)
=
\mathcal{N}
\left(
x_{t-1};
\tilde{\mu}_t(x_t,x_0),
\tilde{\beta}_t I
\right).
$$

方差为：

$$
\tilde{\beta}_t
=
\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.
$$

均值可以写成 $x_t$ 和 $x_0$ 的加权和：

$$
\tilde{\mu}_t(x_t,x_0)
=
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t
+
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0.
$$

这对应 `clip_x0=True` 代码路径：

```python
mean = self.coef1[t] * x_t + self.coef2[t] * x_0
```

其中：

$$
\texttt{coef1}
=
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t},
\qquad
\texttt{coef2}
=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}.
$$

同一个均值也可以写成噪声预测形式：

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right).
$$

这对应 `clip_x0=False` 分支。

## 5. 训练目标

DDPM 原始目标来自变分下界。关键项是在每个时间步比较：

$$
q(x_{t-1}\mid x_t,x_0)
\quad\text{和}\quad
p_\theta(x_{t-1}\mid x_t).
$$

当方差固定时，高斯 KL 项主要比较两个均值。把均值写成噪声预测形式后，目标会变成真实噪声和预测噪声之间的加权 MSE。DDPM 实践中常用简化目标：

$$
\mathcal{L}_{\mathrm{simple}}
=
\mathbb{E}_{x_0,\epsilon,t}
\left[
\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2
\right].
$$

代码中对应：

```python
loss = F.mse_loss(eps_theta, eps)
```

被省略的时间步权重属于论文中的实践简化，它依赖概率目标和建模选择，不只是代数恒等式，因此不在 Lean 中形式化。

## 6. 边界情况

后验方差为：

$$
\tilde{\beta}_t
=
\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.
$$

最后一步逆向采样时约定 $\bar\alpha_{t-1}=1$，因此：

$$
1-\bar\alpha_{t-1}=0
$$

后验方差为 $0$。实现中也在 `t == 0` 时不再添加随机噪声。

## 7. Lean 已验证的恒等式

同目录的 `ddpm.lean` 验证：

- $\alpha=1-\beta$ 可以推出 $1-\alpha=\beta$。
- 如果 $x_t=s+n$，那么 $x_t-n=s$；这是从前向闭式公式反解信号项的代数核心。
- 如果 $1-\bar\alpha_{t-1}=0$，后验方差的分子为 $0$。

Lean 文件不试图形式化高斯条件分布或完整 VLB 分解。这些结论来自 DDPM 和非平衡热力学扩散模型论文；本地 Lean 校验只覆盖和代码直接相关的代数部分。
