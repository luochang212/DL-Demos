# Style Transfer 公式推导

网页教程负责保持可读性；这里补上 Gatys et al. 神经风格迁移的完整符号推导。

## 1. 符号约定

| 符号 | 含义 | 代码变量 |
|------|------|---------|
| $\vec{c}$ | 内容图像 | `content_img` |
| $\vec{s}$ | 风格图像 | `style_img` |
| $\vec{x}$ | 生成图像（优化目标） | `input_img` |
| $F^l$ | VGG 第 $l$ 层特征图，形状 $(C_l, H_l \times W_l)$ | layer output |
| $G^l$ | 第 $l$ 层 Gram 矩阵 | `gram()` return |
| $\alpha$ | 内容损失权重 | `content_weight` |
| $\beta$ | 风格损失权重 | `style_weight` |

## 2. 内容损失 (Content Loss)

内容损失度量生成图像与内容图像在 VGG 高层特征的差异：

$$\mathcal{L}_{\text{content}}(\vec{x}, \vec{c}, l) = \frac{1}{2} \sum_{i,j} \left(F_{ij}^l(\vec{x}) - F_{ij}^l(\vec{c})\right)^2$$

对应代码（`code/style_transfer.py`）：

```python
class ContentLoss(nn.Module):
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
```

## 3. Gram 矩阵

Gram 矩阵捕获特征图各通道之间的相关性（即"风格"）：

$$G_{ij}^l = \sum_k F_{ik}^l \cdot F_{jk}^l$$

矩阵形式：$G^l = F^l \cdot (F^l)^T$，其中 $F^l$ 形状为 $(C_l, H_l \times W_l)$。

Gram 矩阵是对称的：$G_{ij}^l = G_{ji}^l$。

对应代码：

```python
def gram(input):
    a, b, c, d = input.size()  # N, C, H, W
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)
```

## 4. 风格损失 (Style Loss)

风格损失度量生成图像与风格图像在多个 VGG 层的 Gram 矩阵差异：

$$\mathcal{L}_{\text{style}}(\vec{x}, \vec{s}) = \sum_l w_l \cdot \frac{1}{4N_l^2 M_l^2} \sum_{i,j} \left(G_{ij}^l(\vec{x}) - G_{ij}^l(\vec{s})\right)^2$$

其中 $N_l$ 是通道数，$M_l$ 是空间尺寸（$H_l \times W_l$）。

对应代码：

```python
class StyleLoss(nn.Module):
    def forward(self, input):
        G = gram(input)
        self.loss = F.mse_loss(G, self.target)
        return input
```

## 5. 总损失

$$\mathcal{L}_{\text{total}}(\vec{x}, \vec{c}, \vec{s}) = \alpha \cdot \mathcal{L}_{\text{content}}(\vec{x}, \vec{c}) + \beta \cdot \mathcal{L}_{\text{style}}(\vec{x}, \vec{s})$$

当 $\alpha = 1, \beta = 1$ 时：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{content}} + \mathcal{L}_{\text{style}}$。

## 6. Gram 矩阵的对称性

由于 $G = F \cdot F^T$，Gram 矩阵自动对称：

$$G_{ij} = \sum_k F_{ik} F_{jk} = \sum_k F_{jk} F_{ik} = G_{ji}$$

这保证了风格度量对通道顺序不敏感。

## Lean 已验证的恒等式

`derivations/style_transfer.lean` 检查以下代码级代数恒等式：

- Gram 矩阵对称性（代码级恒等）
- 内容损失加法的结合性
- 总损失为单位权重时的直接求和

Lean 文件不试图形式化 VGG 特征提取或图像优化过程。
