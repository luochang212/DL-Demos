import Std

namespace DLDemos.CV.BasicCNN

/-!
Lean checks for algebraic identities used in the basic CNN convolution derivation.

These verify code-facing algebra: output dimension calculation, gradient
accumulation across spatial positions, and group-channel arithmetic.
-/

/-- Output size formula: for H=32, p=1, F=3, s=1, the sliding-window
    count is (32+2*1-3)/1 + 1 = 32. -/
theorem output_size_example (H p F s : Int)
    (hH : H = 32) (hp : p = 1) (hF : F = 3) (hs : s = 1) :
    (H + 2 * p - F) / s + 1 = 32 := by
  rw [hH, hp, hF, hs]
  omega

/-- Gradient accumulation is commutative: accumulating contributions
    across spatial positions (i_h, i_w) into dW is independent of order. -/
theorem gradient_accumulation_commutative (dW_old contrib_a contrib_b : Int) :
    (dW_old + contrib_a) + contrib_b = (dW_old + contrib_b) + contrib_a := by
  omega

/-- When groups=1, each output channel group covers all input channels:
    c_o / 1 = c_o. -/
theorem group_channel_identity (c_o : Int) (hpos : c_o ≠ 0) :
    c_o / (1 : Int) = c_o := by
  exact Int.ediv_eq_of_eq_mul_right (by omega) (by omega)

end DLDemos.CV.BasicCNN
