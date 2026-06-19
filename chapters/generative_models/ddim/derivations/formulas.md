# DDIM 公式推导

本文件补充教程中的公式细节。教程正文优先保证可读性；这里给希望继续推导的人一个完整、但不过度展开的版本。

## 1. 与 DDPM 共用的前向过程

DDIM 不改变训练时的前向扩散过程。给定噪声日程：

$$
\beta_t \in (0, 1),\quad
\alpha_t = 1 - \beta_t,\quad
\bar\alpha_t = \prod_{s=1}^{t}\alpha_s
$$

DDPM 的闭式前向采样为：

$$
q(x_t \mid x_0)
= \mathcal{N}\left(\sqrt{\bar\alpha_t}x_0,\ (1-\bar\alpha_t)I\right)
$$

因此：

$$
x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,
\quad \epsilon \sim \mathcal{N}(0, I)
$$

训练目标仍然是让网络预测噪声：

$$
L_{\text{simple}}
=
\mathbb{E}_{x_0,t,\epsilon}
\left[
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|_2^2
\right]
$$

所以一个已经训练好的 DDPM 噪声预测网络可以直接用于 DDIM 采样。

## 2. 从噪声预测恢复 $x_0$

由前向闭式：

$$
x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon
$$

把真实噪声替换为网络预测噪声 $\epsilon_\theta(x_t,t)$，得到对干净样本的估计：

$$
\hat{x}_0(x_t,t)
=
\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}
{\sqrt{\bar\alpha_t}}
$$

这一步是 DDPM 和 DDIM 采样公式的共同核心。

## 3. DDIM 的跳步更新

设 DDIM 采样子序列为：

$$
\tau_1 < \tau_2 < \cdots < \tau_S
$$

反向采样从 $x_{\tau_i}$ 跳到 $x_{\tau_{i-1}}$。为了简化记号，令：

$$
t = \tau_i,\quad s = \tau_{i-1},\quad s < t
$$

DDIM 把 $x_s$ 写成三项：

$$
x_s
=
\sqrt{\bar\alpha_s}\hat{x}_0
+
\sqrt{1-\bar\alpha_s-\sigma_t^2}\epsilon_\theta(x_t,t)
+
\sigma_t z,
\quad z \sim \mathcal{N}(0,I)
$$

其中：

$$
\sigma_t
=
\eta
\sqrt{
\frac{1-\bar\alpha_s}{1-\bar\alpha_t}
\left(
1-\frac{\bar\alpha_t}{\bar\alpha_s}
\right)
}
$$

将 $\hat{x}_0$ 展开：

$$
\sqrt{\bar\alpha_s}\hat{x}_0
=
\sqrt{\bar\alpha_s}
\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta}{\sqrt{\bar\alpha_t}}
$$

得到实现里使用的三项形式：

$$
x_s
=
\sqrt{\frac{\bar\alpha_s}{\bar\alpha_t}}x_t
+
\left(
\sqrt{1-\bar\alpha_s-\sigma_t^2}
-
\sqrt{\frac{\bar\alpha_s(1-\bar\alpha_t)}{\bar\alpha_t}}
\right)
\epsilon_\theta
+
\sigma_t z
$$

对应代码中的：

```python
first_term = (ab_prev / ab_cur) ** 0.5 * x
second_term = (
    (1 - ab_prev - var) ** 0.5
    - (ab_prev * (1 - ab_cur) / ab_cur) ** 0.5
) * eps
third_term = var**0.5 * noise
x = first_term + second_term + third_term
```

其中 `ab_cur` 是 $\bar\alpha_t$，`ab_prev` 是 $\bar\alpha_s$，`var` 是 $\sigma_t^2$。

## 4. $\eta$ 的含义

当 $\eta=0$：

$$
\sigma_t = 0
$$

于是：

$$
x_s
=
\sqrt{\bar\alpha_s}\hat{x}_0
+
\sqrt{1-\bar\alpha_s}\epsilon_\theta
$$

采样过程没有额外随机噪声。若初始 $x_T$ 相同，输出也相同，因此它是确定性采样。

当 $\eta=1$ 且子序列使用完整相邻时间步时，DDIM 的方差项退化到 DDPM 论文中的后验方差形式。若使用跳步采样，即使 $\eta=1$，它也不是原始 1000 步 DDPM 的逐步马尔可夫采样。

## 5. Lean4 验证范围

Lean4 文件验证本推导中最容易写错的基础代数关系：

- $\eta=0$ 时 $\sigma_t^2=0$；
- $\hat{x}_0$ 代回后，第一项和噪声修正项的系数拆分；
- `var` 与 $\sigma_t^2$ 的平方关系。

概率分布、极限过程和神经网络近似误差不在 Lean4 文件中形式化。
