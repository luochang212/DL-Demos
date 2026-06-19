# Basic CNN 公式推导

网页教程负责保持可读性；这里补上 2D 卷积前向与反向传播的完整符号推导。

## 1. 符号约定

| 符号 | 含义 | 代码变量 |
|------|------|---------|
| $H, W, C_i$ | 输入高度、宽度、通道数 | `h_i, w_i, c_i` |
| $C_o$ | 输出通道数 | `c_o` |
| $F$ | 卷积核空间尺寸 | `f` |
| $s$ | 步幅 (stride) | `stride` |
| $p$ | 填充 (padding) | `padding` |
| $d$ | 膨胀 (dilation) | `dilation` |
| $g$ | 分组数 (groups) | `groups` |
| $A$ | 输入激活 | `input` |
| $W$ | 卷积核权重 | `weight` |
| $b$ | 偏置 | `bias` |
| $Z$ | 卷积输出（激活前） | `output` |

## 2. 输出尺寸计算

带 padding 和 stride 的卷积输出尺寸：

$$H_{\text{out}} = \left\lfloor \frac{H + 2p - F}{s} \right\rfloor + 1$$

$$W_{\text{out}} = \left\lfloor \frac{W + 2p - F}{s} \right\rfloor + 1$$

带 dilation 时，有效卷积核尺寸：

$$F_{\text{new}} = F + (F - 1) \cdot (d - 1)$$

对应代码（`code/np_conv.py`）：

```python
def cal_new_sidelngth(sl, s, f, p):
    return (sl + 2 * p - f) // s + 1

f_new = f + (f - 1) * (dilation - 1)
h_o = cal_new_sidelngth(h_i, stride, f_new, padding)
w_o = cal_new_sidelngth(w_i, stride, f_new, padding)
```

## 3. 2D 卷积前向传播

对于输出位置 $(h, w)$ 和输出通道 $c_o$（属于分组 $g = \lfloor c_o / C_{\text{per\_group}} \rfloor$）：

$$Z[h, w, c_o] = \sum_{c_k} \sum_{i=0}^{F_{\text{new}}-1} \sum_{j=0}^{F_{\text{new}}-1} A_{\text{pad}}[h \cdot s + i, w \cdot s + j, g \cdot C_k + c_k] \cdot W_{\text{new}}[c_o, i, j, c_k] + b[c_o]$$

其中 $W_{\text{new}}$ 是 dilation 扩展后的权重：

$$W_{\text{new}}[c_o, i \cdot d, j \cdot d, c_k] = W[c_o, i, j, c_k]$$

对应代码（`code/np_conv.py`）：

```python
for i_h in range(h_o):
    for i_w in range(w_o):
        for i_c in range(c_o):
            i_g = i_c // c_o_per_group
            h_lower = i_h * stride
            h_upper = i_h * stride + f_new
            w_lower = i_w * stride
            w_upper = i_w * stride + f_new
            c_lower = i_g * c_k
            c_upper = (i_g + 1) * c_k
            input_slice = input_pad[h_lower:h_upper, w_lower:w_upper, c_lower:c_upper]
            kernel_slice = weight_new[i_c]
            output[i_h, i_w, i_c] = np.sum(input_slice * kernel_slice)
            if bias is not None:
                output[i_h, i_w, i_c] += bias[i_c]
```

## 4. 卷积反向传播

### 4.1 对权重的梯度 (dW)

每个空间位置对 $dW$ 的贡献：

$$dW[c_o, :, :, c_k] += A_{\text{pad}}[\text{slice}] \cdot dZ[h, w, c_o]$$

### 4.2 对输入的梯度 (dA_prev)

梯度通过卷积核反向传播：

$$dA_{\text{pad}}[\text{slice}] += W[c_o, :, :, c_k] \cdot dZ[h, w, c_o]$$

### 4.3 对偏置的梯度 (db)

$$db[c_o] += dZ[h, w, c_o]$$

对应代码（`code/np_conv_backward.py`）：

```python
for i_h in range(h_o):
    for i_w in range(w_o):
        for i_c in range(c_o):
            input_slice = A_prev_pad[h_lower:h_upper, w_lower:w_upper, :]
            dW[i_c] += input_slice * dZ[i_h, i_w, i_c]
            dA_prev_pad[h_lower:h_upper, w_lower:w_upper, :] += (
                W[i_c] * dZ[i_h, i_w, i_c]
            )
            db[i_c] += dZ[i_h, i_w, i_c]
```

## 5. 梯度累加的交换性

由于 $dW$ 和 $dA_{\text{prev}}$ 通过空间循环累加，累加顺序不影响最终结果（加法交换律）：

$$(dW_{\text{old}} + \text{contrib}_a) + \text{contrib}_b = (dW_{\text{old}} + \text{contrib}_b) + \text{contrib}_a$$

这保证了无论以何种顺序遍历 $(i_h, i_w)$，梯度结果一致。

## Lean 已验证的恒等式

`derivations/basic_cnn.lean` 检查以下代码级代数恒等式：

- 输出尺寸公式（以 H=32, p=1, F=3, s=1 为例）
- 梯度累加的交换性
- groups=1 时通道除法的恒等性

Lean 文件不试图形式化卷积的连续数学性质或自动求导框架。
它只检查代码中关键整数算术恒等式。
