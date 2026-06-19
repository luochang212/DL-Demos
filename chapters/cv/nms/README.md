# IoU 与 NMS — 目标检测后处理

IoU（交并比）计算与 NMS（非极大值抑制）算法教程。覆盖边界框交并计算、
IoU 公式推导、以及贪婪 NMS 算法的从零实现。

## 目录结构

- `code/` — 规范实现
  - `code/iou.py` — 边界框面积、交集、IoU 计算
  - `code/nms.py` — NMS 算法 + 可视化渲染（`BoxRenderer`）
  - `code/show_bbox.py` — PIL 边界框绘制工具
- `derivations/` — 完整公式推导与 Lean 形式化验证
  - `derivations/formulas.md` — 面积、交集、IoU、NMS 算法的完整符号推导
  - `derivations/nms.lean` — IoU 对称性、面积非负、分数过滤
- `tests/` — CPU 烟雾测试
- `bboxes.pt` — 预计算的检测结果示例（用于可视化演示）

## 运行命令

### 烟雾测试

```bash
uv run pytest chapters/cv/nms/tests -q
```

### IoU 可视化演示

```bash
uv run python -m chapters.cv.nms.code.iou
```

### NMS 可视化演示

```bash
uv run python -m chapters.cv.nms.code.nms
```

### 代码入口

```bash
uv run python -c "
from chapters.cv.nms.code.iou import iou
from chapters.cv.nms.code.nms import nms
print('IoU and NMS imported')
"
```

## 数据与依赖

- 无外部数据依赖。`bboxes.pt` 是预生成的示例检测结果。
- 可视化需要 `work_dirs/detection_demo.jpg`（运行时下载或生成）。
- 依赖 `numpy`、`Pillow`（可视化时需要）。

## 输出位置

- 可视化输出到 `work_dirs/nms/`。

## 参考资料

- Felzenszwalb, P. F., Girshick, R. B., McAllester, D., & Ramanan, D. (2010). [Object Detection with Discriminatively Trained Part-Based Models](https://ieeexplore.ieee.org/document/5255236).
- Bodla, N., Singh, B., Chellappa, R., & Davis, L. S. (2017). [Soft-NMS — Improving Object Detection With One Line of Code](https://arxiv.org/abs/1704.04503).
- Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R., & Ren, D. (2020). [Distance-IoU Loss](https://arxiv.org/abs/1911.08287).
- 完整公式推导：[derivations/formulas.md](derivations/formulas.md)
