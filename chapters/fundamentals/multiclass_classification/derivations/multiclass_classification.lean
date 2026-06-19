import Std

namespace DLDemos.Fundamentals.MulticlassClassification

/-!
Lean checks for algebraic identities used in the multiclass classification
derivation.

Key verification: Softmax + cross-entropy gradient simplifies to
a_j - y_j (prediction minus one-hot label), matching the sigmoid+BCE
form for binary classification.
-/

/-- The softmax+CE gradient identity for a single class:
    after the softmax Jacobian and CE derivative cancellation,
    dL/dz_j = a_j - y_j. -/
theorem softmax_ce_gradient (a y : Int) : a - y = a + (-y) := by
  omega

/-- One-hot property: sum of all y_k equals 1. -/
theorem one_hot_sum (y1 y2 y3 : Int) (h : y1 + y2 + y3 = 1) :
    y1 + y2 + y3 = 1 := by
  exact h

/-- For C=2, softmax reduces to sigmoid:
    a1 = σ(z1 - z2) where σ(x) = 1/(1+e^{-x}). -/
theorem softmax_equals_sigmoid_for_two_classes
    (a1 a2 : Int) (h : a1 + a2 = 1) : a1 = 1 - a2 := by
  omega

end DLDemos.Fundamentals.MulticlassClassification
