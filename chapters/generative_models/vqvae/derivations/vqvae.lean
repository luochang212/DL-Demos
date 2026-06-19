import Std

namespace DLDemos.GenerativeModels.VQVAE

/-!
Lean checks for simple algebra used by the VQVAE derivation.
-/

theorem straight_through_forward_value (ze zq : Int) :
    ze + (zq - ze) = zq := by
  omega

theorem three_loss_terms_reorder (reconstruct embedding commitment : Int) :
    reconstruct + embedding + commitment
      = embedding + reconstruct + commitment := by
  omega

end DLDemos.GenerativeModels.VQVAE
