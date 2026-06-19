# Advanced Optimizer 公式推导

网页教程负责保持可读性；这里补上 SGD、Momentum、RMSProp、Adam 四种
优化器的完整符号推导。

## 1. 符号约定

| 符号 | 含义 | 代码变量 |
|------|------|---------|
| $\theta$ | 模型参数 | `self.param_dict[k]` |
| $g_t$ | 第 $t$ 步梯度 | `self.grad_dict[k]` |
| $\eta$ | 学习率 | `self.learning_rate` |
| $\beta$ | Momentum / RMSProp 衰减系数 | `self.beta` |
| $\beta_1$ | Adam 一阶矩衰减系数 | `self.beta1` |
| $\beta_2$ | Adam 二阶矩衰减系数 | `self.beta2` |
| $\epsilon$ | 数值稳定常数 | `self.eps` |
| $v_t$ | 一阶矩估计 (Momentum / Adam) | `self.v_dict[k]` |
| $s_t$ | 二阶矩估计 (RMSProp / Adam) | `self.s_dict[k]` |
| $t$ | 当前步数 | `self._num_step` |

## 2. Gradient Descent (SGD)

最基础的优化器，参数沿负梯度方向更新：

$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$

对应代码（`code/optimizer.py`）：

```python
self.param_dict[k] -= self.learning_rate * self.grad_dict[k]
```

**问题**：在峡谷形损失面上震荡，收敛慢。

## 3. Momentum

引入速度 $v_t$（梯度的指数移动平均），累积历史梯度方向：

$$v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot g_t$$

$$\theta_{t+1} = \theta_t - \eta \cdot v_t$$

展开 $v_t$：

$$v_t = (1-\beta) \sum_{i=0}^{t-1} \beta^i \cdot g_{t-i}$$

对应代码：

```python
self.velocity_dict[k] = (
    self.beta * self.velocity_dict[k] + (1 - self.beta) * self.grad_dict[k]
)
self.param_dict[k] -= self.learning_rate * self.velocity_dict[k]
```

**直觉**：$\beta$ 越大（如 0.9），动量越持久，越能穿越平坦区域和局部极小。

## 4. RMSProp

自适应学习率：用梯度平方的 EMA 缩放每个参数的更新步长：

$$s_t = \beta \cdot s_{t-1} + (1 - \beta) \cdot g_t^2$$

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{g_t}{\sqrt{s_t} + \epsilon}$$

**偏差修正**（可选）：

$$\hat{s}_t = \frac{s_t}{1 - \beta^t}$$

对应代码：

```python
self.s_dict[k] = self.beta * self.s_dict[k] + (1 - self.beta) * np.square(
    self.grad_dict[k]
)
if self.correct_param:
    s = self.s_dict[k] / (1 - self.beta**self._num_step)
else:
    s = self.s_dict[k]
self.param_dict[k] -= (
    self.learning_rate * self.grad_dict[k] / (np.sqrt(s + self.eps))
)
```

## 5. Adam

结合 Momentum（一阶矩）和 RMSProp（二阶矩），是目前最常用的优化器：

### 5.1 一阶矩估计（动量）

$$v_t = \beta_1 \cdot v_{t-1} + (1 - \beta_1) \cdot g_t$$

### 5.2 二阶矩估计（自适应学习率）

$$s_t = \beta_2 \cdot s_{t-1} + (1 - \beta_2) \cdot g_t^2$$

### 5.3 偏差修正

由于 $v_0 = 0$ 和 $s_0 = 0$，初期估计偏向零。Adam 使用偏差修正：

$$\hat{v}_t = \frac{v_t}{1 - \beta_1^t}$$

$$\hat{s}_t = \frac{s_t}{1 - \beta_2^t}$$

### 5.4 参数更新

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{v}_t}{\sqrt{\hat{s}_t} + \epsilon}$$

### 5.5 特殊情况：t = 1 时的偏差修正

当 $t = 1$ 时，$v_1 = (1-\beta_1) \cdot g_1$，$\hat{v}_1 = v_1 / (1-\beta_1^1) = g_1$。

即第一步的修正后一阶矩等于当前梯度 — 偏差修正消除了初始零偏差。

对应代码：

```python
self.v_dict[k] = (
    self.beta1 * self.v_dict[k] + (1 - self.beta1) * self.grad_dict[k]
)
self.s_dict[k] = self.beta2 * self.s_dict[k] + (1 - self.beta2) * (
    self.grad_dict[k] ** 2
)
if self.correct_param:
    v = self.v_dict[k] / (1 - self.beta1**self._num_step)
    s = self.s_dict[k] / (1 - self.beta2**self._num_step)
else:
    v = self.v_dict[k]
    s = self.s_dict[k]
self.param_dict[k] -= self.learning_rate * v / (np.sqrt(s) + self.eps)
```

### 5.6 Adam 的超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| $\beta_1$ | 0.9 | 一阶矩衰减率 |
| $\beta_2$ | 0.999 | 二阶矩衰减率 |
| $\epsilon$ | $10^{-8}$ | 数值稳定 |
| $\eta$ | 可变 | 学习率 |

## 6. 学习率调度

代码实现了双曲线衰减调度器：

$$\eta_t = \frac{\eta_0}{1 + t \cdot \text{decay\_rate}}$$

对应代码：

```python
def scheduler(learning_rate_zero, epoch):
    return learning_rate_zero / (1 + epoch * decay_rate)
```

## Lean 已验证的恒等式

`derivations/advanced_optimizer.lean` 检查以下代码级代数恒等式：

- Adam 偏差修正（t=1 时 $\hat{v}_1 = g_1$）
- Momentum 速度递推的代数等价形式
- RMSProp 分母中 $\sqrt{s} + \epsilon$ 的正性（整数近似）

Lean 文件不试图形式化优化理论（收敛性、regret bound 等）。
它只检查代码中优化器更新步骤的关键整数比例关系。
