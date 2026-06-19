import numpy as np

from chapters.cv.nms.code.iou import area, box_intersection, iou
from chapters.cv.nms.code.nms import nms


def test_iou_area_and_intersection():
    assert area((0, 0, 10, 10)) == 100
    assert box_intersection((0, 0, 10, 10), (5, 5, 15, 15)) == (5, 5, 10, 10)
    # identical boxes -> 1.0
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    # disjoint boxes -> 0.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # partial overlap: intersection 25, union 175
    assert abs(iou((0, 0, 10, 10), (5, 5, 15, 15)) - 25 / 175) < 1e-9


def test_nms_suppresses_overlapping_boxes():
    # each row: [probability, x1, y1, x2, y2]
    predicts = np.array(
        [
            [0.9, 0, 0, 10, 10],  # highest score, kept
            [0.8, 1, 1, 11, 11],  # overlaps box 0 -> suppressed
            [0.7, 50, 50, 60, 60],  # disjoint -> kept
            [0.1, 0, 0, 5, 5],  # below score_thresh -> filtered
        ],
        dtype=float,
    )
    _, indices = nms(predicts, score_thresh=0.5, iou_thresh=0.3)
    assert indices == [0, 2]
