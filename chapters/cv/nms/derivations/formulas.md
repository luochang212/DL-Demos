# IoU 与 NMS 公式推导

网页教程负责保持可读性；这里补上 IoU 计算和 NMS 算法的完整符号推导。

## 1. 符号约定

| 符号 | 含义 | 代码变量 |
|------|------|---------|
| $(x_1, y_1, x_2, y_2)$ | 边界框（左上角 + 右下角） | bounding box |
| $w, h$ | 边界框宽度、高度 | `w = max(x2 - x1, 0)` |
| $A, B$ | 两个边界框 | `box_a`, `box_b` |
| $I$ | 交集区域 | `box_intersection()` return |
| $\text{score}$ | 检测置信度 | score in `bboxes` |
| $\tau$ | IoU 阈值 | `iou_threshold` |

## 2. 边界框面积

$$A = w \times h, \quad w = \max(x_2 - x_1, 0), \quad h = \max(y_2 - y_1, 0)$$

面积非负：$A \ge 0$（当 $w \ge 0, h \ge 0$）。

对应代码（`code/iou.py`）：

```python
def area(box):
    return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])
```

## 3. 边界框交集 (Box Intersection)

两个边界框的交集区域：

$$x_1^{\text{inter}} = \max(x_1^A, x_1^B)$$
$$y_1^{\text{inter}} = \max(y_1^A, y_1^B)$$
$$x_2^{\text{inter}} = \min(x_2^A, x_2^B)$$
$$y_2^{\text{inter}} = \min(y_2^A, y_2^B)$$

交集面积为：

$$I = \max(x_2^{\text{inter}} - x_1^{\text{inter}}, 0) \times \max(y_2^{\text{inter}} - y_1^{\text{inter}}, 0)$$

对应代码：

```python
def box_intersection(box_a, box_b):
    x1 = torch.max(box_a[..., 0], box_b[..., 0])
    y1 = torch.max(box_a[..., 1], box_b[..., 1])
    x2 = torch.min(box_a[..., 2], box_b[..., 2])
    y2 = torch.min(box_a[..., 3], box_b[..., 3])
    return torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
```

## 4. IoU (Intersection over Union)

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{I}{\text{area}(A) + \text{area}(B) - I}$$

**性质**：
- $\text{IoU} \in [0, 1]$
- 对称性：$\text{IoU}(A, B) = \text{IoU}(B, A)$（因为交集和并集都是对称的）
- $\text{IoU} = 0$：无重叠
- $\text{IoU} = 1$：完全重叠

对应代码：

```python
def iou(box_a, box_b):
    inter = box_intersection(box_a, box_b)
    area_a = area(box_a)
    area_b = area(box_b)
    union = area_a + area_b - inter
    return inter / union
```

## 5. NMS 算法

Non-Maximum Suppression 按置信度降序处理检测框：

1. 按 score 降序排序
2. 过滤 score < threshold 的框
3. 选择最高分框，加入结果集
4. 移除与已选框 IoU > $\tau$ 的所有框
5. 重复步骤 3-4 直到候选集为空

对应代码（`code/nms.py`）：

```python
def nms(bboxes, scores, score_threshold=0.5, iou_threshold=0.5):
    assert len(bboxes) > 0
    bboxes = bboxes[scores > score_threshold]
    scores = scores[scores > score_threshold]
    keep = []
    order = np.argsort(scores)[::-1]
    while len(order) > 0:
        keep.append(order[0])
        ious = iou(bboxes[order[0]], bboxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return bboxes[keep], keep
```

## 6. 分数阈值过滤

当 $\text{score} < \tau_{\text{score}}$ 时，框被直接丢弃：

$$\text{keep} = \{i \mid \text{scores}[i] > \tau_{\text{score}}\}$$

## Lean 已验证的恒等式

`derivations/nms.lean` 检查以下代码级代数恒等式：

- IoU 对称性（交集面积的恒等性）
- 面积计算示例（w=10, h=10 → A=100）
- 分数过滤谓词

Lean 文件不试图形式化 NMS 的算法复杂度或 IoU 的连续性质。
