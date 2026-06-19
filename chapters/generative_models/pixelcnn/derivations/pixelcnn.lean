import Std

namespace DLDemos.GenerativeModels.PixelCNN

/-!
Lean checks for simple algebra used by the PixelCNN derivation.
-/

theorem nll_two_pixels_reorder (a b : Int) :
    -(a + b) = -b - a := by
  omega

theorem mean_cross_entropy_numerator_reorder (a b : Int) :
    a + b = b + a := by
  omega

end DLDemos.GenerativeModels.PixelCNN
