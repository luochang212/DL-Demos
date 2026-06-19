import Std

namespace DLDemos.Fundamentals.LogisticRegression

/-!
Lean checks for algebraic identities used in the logistic regression derivation.

The calculus part (sigmoid derivative, chain rule) is documented in `formulas.md`.
This file verifies the simple algebra after the calculus steps are accepted:

1. BCE + sigmoid gradient cancellation: `(1-a)(-y) + a(1-y) = a - y`

Where `a` represents sigmoid(z) and `y` is the binary label (0 or 1).
-/

theorem sigmoid_bce_cancel (a y : Int) :
    (1 - a) * (-y) + a * (1 - y) = a - y := by
  calc
    (1 - a) * (-y) + a * (1 - y)
        = ((1 - a) * (-y)) + (a * 1 - a * y) := by
      rw [Int.mul_sub]
    _ = ((1 - a) * (-y)) + (a - a * y) := by simp
    _ = (1 * (-y) - a * (-y)) + (a - a * y) := by
      rw [Int.sub_mul]
    _ = (-y - a * (-y)) + (a - a * y) := by simp
    _ = (-y - (-(a * y))) + (a - a * y) := by
      rw [Int.mul_neg]
    _ = (-y + a * y) + (a - a * y) := by simp
    _ = a - y := by omega

/-- The gradient simplifies to prediction minus label, regardless of the
    intermediate sigmoid value.  This is the key computational advantage of
    pairing sigmoid with binary cross-entropy. -/
theorem gradient_simplifies_to_prediction_minus_label (a y : Int) :
    (1 - a) * (-y) + a * (1 - y) = a - y :=
  sigmoid_bce_cancel a y

/-- Expanding the right side: a*y - a*y cancels out. -/
theorem cancellation_of_cross_terms (a y : Int) :
    (-y + a * y) + (a - a * y) = a - y := by
  omega

end DLDemos.Fundamentals.LogisticRegression
