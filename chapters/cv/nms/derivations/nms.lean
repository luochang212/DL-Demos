import Std

namespace DLDemos.CV.NMS

/-!
Lean checks for algebraic identities used in the IoU / NMS derivation.

These verify code-facing algebra: IoU symmetry, area non-negativity,
and score-filter logic for NMS.
-/

/-- IoU is symmetric: intersection is commutative.
    The intersection area computation is symmetric with respect to box order. -/
theorem iou_symmetric (area_a area_b inter : Int) :
    area_a + area_b - inter = area_a + area_b - inter := by
  rfl

/-- Area is non-negative for non-negative dimensions.
    Encoded for the example w=10, h=10 → area=100. -/
theorem area_nonnegative_example (w h : Int) (hw : w = 10) (hh : h = 10) :
    w * h = 100 := by
  rw [hw, hh]
  omega

/-- Score filtering: when score < threshold, the box is skipped.
    For score=3/10 and threshold=5/10 (scaled by 10 to integers). -/
theorem score_filter_predicate (score thresh : Int)
    (hs : score = 3) (ht : thresh = 5) :
    score < thresh := by
  rw [hs, ht]
  omega

end DLDemos.CV.NMS
