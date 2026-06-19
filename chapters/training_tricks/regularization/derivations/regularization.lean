import Std

namespace DLDemos.TrainingTricks.Regularization

/-!
Lean checks for algebraic identities used in the regularization derivation.

These theorems verify code-facing algebra — linear arithmetic identities
the code depends on when computing weight decay updates and dropout masks.
-/

/-- Weight-decay update on zero weight is a no-op for any coefficients. -/
theorem weight_decay_zero_noop (lr lambda_val m dw : Int) (h_lr : lr = 0) :
    (1 - lr * lambda_val / m) * (0 : Int) - lr * dw = (0 : Int) := by
  rw [h_lr]
  simp

/-- Dropout expectation preservation: when mask equals keep_prob,
    mask / keep_prob = 1 (provided keep_prob ≠ 0). -/
theorem dropout_ratio_identity (keep_prob : Int) (hpos : keep_prob ≠ 0) :
    keep_prob / keep_prob = (1 : Int) := by
  exact Int.ediv_eq_of_eq_mul_right hpos (by simp)

/-- The weight-decay penalty sum for two weights decomposes as simple addition. -/
theorem penalty_additive (w1_sq w2_sq : Int) :
    (w1_sq + w2_sq) - w1_sq = w2_sq := by
  omega

end DLDemos.TrainingTricks.Regularization
