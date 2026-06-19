import Std

namespace DLDemos.GenerativeModels.VAE

/-!
Lean checks for algebraic identities used in the VAE derivation.

These theorems intentionally avoid formalizing probability theory. They verify
the code-facing algebra after the probabilistic KL result is accepted.
-/

theorem kl_code_form_matches_canonical_unscaled
    (meanSq var logvar : Int) :
    -(1 + logvar - meanSq - var) = meanSq + var - logvar - 1 := by
  omega

theorem kl_code_form_matches_canonical_scaled_by_two
    (meanSq var logvar : Int) :
    2 * (-(1 + logvar - meanSq - var))
      = 2 * (meanSq + var - logvar - 1) := by
  omega

end DLDemos.GenerativeModels.VAE
