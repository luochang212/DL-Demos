# ResNet 公式推导

网页教程负责保持可读性；这里补上残差连接和瓶颈块的完整符号推导。

## 1. 符号约定

| 符号 | 含义 | 代码变量 |
|------|------|---------|
| $x$ | 残差块输入 | `input_tensor` |
| $F(x, \{W_i\})$ | 残差函数（学习的映射） | conv → BN → ReLU 路径 |
| $y$ | 残差块输出 | `output` |
| $W_i$ | 第 $i$ 层权重 | 卷积层权重 |
| $W_s$ | shortcut 1×1 卷积权重（维度匹配时） | conv block 的 shortcut 分支 |

## 2. 残差块核心公式

### 2.1 Identity Block（维度不变）

输入输出维度相同，shortcut 为恒等映射：

$$y = F(x, \{W_i\}) + x$$

对应代码（`code/tf_main.py` 中的 `identity_block_2`）：

```python
X = Conv2D(filters=F3, kernel_size=(f, f), strides=(1, 1), padding='same')(X_shortcut)
X = BatchNormalization(axis=3)(X)
X = Activation('relu')(X)

X = Conv2D(filters=F3, kernel_size=(f, f), strides=(1, 1), padding='same')(X)
X = BatchNormalization(axis=3)(X)

X = Add()([X, X_shortcut])
X = Activation('relu')(X)
```

### 2.2 Convolution Block（维度改变）

当 stride > 1 或通道数不匹配时，shortcut 需经过 1×1 卷积：

$$y = F(x, \{W_i\}) + W_s x$$

**特殊情况**：当 $F(x) = 0$（残差未学到任何东西），$y = x$（或 $y = W_s x$），块退化为恒等映射。

## 3. 梯度流分析

残差连接的关键优势：梯度可以通过 shortcut 直接传播。

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(\frac{\partial F}{\partial x} + 1\right)$$

即使 $\partial F / \partial x$ 很小（梯度消失），shortcut 保证梯度至少为 1：

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot 1 \quad (\text{当 } \partial F / \partial x \approx 0)$$

## 4. 瓶颈块 (Bottleneck)

ResNet-50/101/152 使用 3 层瓶颈结构：

1. 1×1 卷积压缩通道：$C_{\text{in}} \to C_{\text{bottleneck}}$
2. 3×3 卷积空间特征提取
3. 1×1 卷积扩展通道：$C_{\text{bottleneck}} \to C_{\text{out}}$

1×1 卷积在 identity block 中保证 $C_{\text{in}} = C_{\text{out}}$，
在 convolution block 中实现 $C_{\text{in}} \to C_{\text{out}}$ 的通道匹配。

## 5. ResNet 变体参数

| 层数 | 结构 | 参数量 |
|------|------|--------|
| ResNet-18 | [2,2,2,2] × 2-layer blocks | 11M |
| ResNet-34 | [3,4,6,3] × 2-layer blocks | 21M |
| ResNet-50 | [3,4,6,3] × 3-layer bottleneck | 25M |
| ResNet-101 | [3,4,23,3] × 3-layer bottleneck | 44M |
| ResNet-152 | [3,8,36,3] × 3-layer bottleneck | 60M |

## Lean 已验证的恒等式

`derivations/resnet.lean` 检查以下代码级代数恒等式：

- Identity shortcut 在零残差时退化为恒等映射：$x + 0 = x$
- Shortcut 梯度非零：$dF + 1 = 1$（当 $dF = 0$）
- 卷积块通道匹配

Lean 文件不试图形式化梯度流分析或网络收敛性。
