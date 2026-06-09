# IoU and Non-Maximum Suppression

Run the unit-tested core implementation:

```shell
uv run pytest tests/test_tutorial_models.py -k DetectionPostprocessing
```

The visualization examples in `iou.py`, `show_bbox.py`, and `nms.py` require
`work_dirs/detection_demo.jpg`. Bounding boxes use `(x1, y1, x2, y2)` coordinates.
The provided NMS implementation is class-agnostic; run it separately per class
when processing multi-class detector outputs.
