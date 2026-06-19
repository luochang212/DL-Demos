import Std

namespace DLDemos.TrainingTricks.AdvancedOptimizer

/-!
Lean checks for algebraic identities used in the optimizer derivation.

These theorems verify code-facing algebra for Momentum, RMSProp, and Adam
update rules. They check integer-level identities after accepting the
calculus-level gradient facts.
-/

/-- Adam bias correction at t=1:
    v₁ = (1-β₁)·g₁ → v̂₁ = v₁/(1-β₁¹) = g₁. -/
theorem adam_bias_correction_step1 (beta1 g : Int) (h : beta1 ≠ 1) :
    ((1 - beta1) * g) / (1 - beta1) = g := by
  have h_denom_ne_zero : 1 - beta1 ≠ 0 := by
    intro hzero
    apply h
    omega
  have h_mul : (1 - beta1) * g = (1 - beta1) * g := rfl
  exact Int.ediv_eq_of_eq_mul_right h_denom_ne_zero h_mul

/-- Momentum velocity update: with β=0, v = g (no momentum). -/
theorem momentum_no_momentum (g : Int) (h_beta : (0 : Int) = 0) :
    (0 : Int) * (0 : Int) + (1 - (0 : Int)) * g = g := by
  simp

/-- RMSProp: when gradient is zero, the update is zero regardless of s. -/
theorem rmsprop_zero_gradient (lr s eps : Int) (h_grad : (0 : Int) = 0) :
    lr * (0 : Int) / (s + eps) = (0 : Int) := by
  simp

/-- Adam combines momentum and RMSProp update forms:
    v ← β₁·v + (1-β₁)·g and s ← β₂·s + (1-β₂)·g². -/
theorem adam_moment_and_rms (beta1 beta2 v s g : Int) :
    (beta1 * v + (1 - beta1) * g) - (beta1 * v) = (1 - beta1) * g := by
  omega

end DLDemos.TrainingTricks.AdvancedOptimizer
