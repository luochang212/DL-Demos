# PixelCNN 公式推导

本文档保存 PixelCNN 章节的完整符号推导。网页教程负责保持可读性；这里补上自回归分解、分类损失和掩码约束背后的公式。

## 1. 符号约定

- $x$：一张灰度图像。
- $x_i$：按光栅扫描顺序展开后的第 $i$ 个像素。
- $x_{<i}$：第 $i$ 个像素之前的所有像素。
- $K$：像素量化等级数，对应代码里的 `color_level`。
- $f_\theta$：PixelCNN 网络。
- $\pi_i=f_\theta(x_{<i})$：第 $i$ 个像素的 $K$ 类概率分布。

## 2. 自回归概率分解

图像的联合概率可以按像素顺序分解：

$$
p_\theta(x)
=
\prod_{i=1}^{n}p_\theta(x_i\mid x_{<i}).
$$

取对数后得到：

$$
\log p_\theta(x)
=
\sum_{i=1}^{n}\log p_\theta(x_i\mid x_{<i}).
$$

训练时最大化这个对数似然，等价于最小化负对数似然：

$$
\mathcal{L}
=
-
\sum_{i=1}^{n}\log p_\theta(x_i\mid x_{<i}).
$$

## 3. 分类交叉熵

当每个像素被量化为 $K$ 个颜色等级时，第 $i$ 个像素是一个分类标签 $y_i\in\{0,\ldots,K-1\}$。网络输出 logits，softmax 后得到概率 $\pi_i$：

$$
p_\theta(x_i=y_i\mid x_{<i})=\pi_{i,y_i}.
$$

因此单像素损失为：

$$
\ell_i=-\log \pi_{i,y_i}.
$$

全图损失为所有像素的平均交叉熵：

$$
\mathcal{L}
=
\frac{1}{n}\sum_{i=1}^{n}-\log \pi_{i,y_i}.
$$

这对应代码中的：

```python
loss = nn.CrossEntropyLoss()(predict_y, y)
```

其中 `predict_y` 形状为 `(N, color_level, H, W)`，`y` 形状为 `(N, H, W)`。

## 4. 掩码约束

PixelCNN 的关键约束是：预测第 $i$ 个像素时不能访问 $x_i$ 之后的像素。

第一层使用 A 型掩码：

$$
f_\theta^{(1)}(x)_i = g(x_{<i})
$$

也就是当前像素自身也不可见。后续层使用 B 型掩码：

$$
f_\theta^{(l)}(h^{(l-1)})_i = g(h^{(l-1)}_{\le i})
$$

B 型掩码允许当前位置特征参与计算，但由于第一层已经阻断了未来像素，后续层不会重新引入未来信息。

Gated PixelCNN 把上下文拆成垂直栈和水平栈。垂直栈负责看上方像素，水平栈负责看左侧像素，垂直栈信息通过 $1\times1$ 卷积注入水平栈，从而减少原始 PixelCNN 的盲点问题。

## 5. Lean 已验证的恒等式

同目录的 `pixelcnn.lean` 验证：

- 由 $p(x)=p_1p_2$ 可得 $-\log p(x)=-(\log p_1+\log p_2)$ 的代数骨架。
- 两个像素负对数似然相加时，项的顺序重排不改变总损失。

Lean 文件不形式化 softmax、概率测度或卷积可见性；本地 Lean 校验只覆盖和损失展开直接相关的代数部分。
