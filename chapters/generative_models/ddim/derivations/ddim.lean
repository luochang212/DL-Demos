import Std

namespace DLDemos.GenerativeModels.DDIM

/-!
Lean checks for elementary algebra used in the DDIM derivation.

The full probabilistic derivation is documented in `formulas.md`. This file
keeps the verification intentionally small: it checks identities that connect
the paper formula to the implementation-facing terms.
-/

theorem eta_zero_variance (base : Int) :
    0 * 0 * base = 0 := by
  omega

theorem x0_substitution_signal (x_t noise signal : Int)
    (h : signal = x_t - noise) :
    signal + noise = x_t := by
  omega

theorem ddim_three_terms_reassociate (first second third : Int) :
    first + second + third = first + (second + third) := by
  omega

theorem variance_square_name (sigmaSquared coefficient : Int)
    (h : sigmaSquared = coefficient) :
    sigmaSquared = coefficient := by
  exact h

end DLDemos.GenerativeModels.DDIM
