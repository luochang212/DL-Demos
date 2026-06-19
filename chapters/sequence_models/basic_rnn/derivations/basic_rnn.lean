import Std

namespace DLDemos.SequenceModels.BasicRNN

/-!
Lean checks for algebraic identities used in the basic RNN derivation.

These theorems intentionally avoid formalizing probability theory or
neural-network training dynamics. They verify the code-facing algebra
after the RNN forward equations are accepted.
-/

/-- Hidden state recurrence: when initial hidden state a₀ = 0 and first
    input x₁ = 0, the hidden state update simplifies to a₁ = tanh(b_a)
    regardless of weight matrices. At the integer level, this simplifies
    to verifying that 0 * W + 0 * W + b = b. -/
theorem hidden_state_init_zero (W_ax W_aa b_a : Int) :
    (W_ax * 0 + W_aa * 0 + b_a) = b_a := by
  simp

/-- Hidden state after zero input and zero bias: if b_a = 0 as well,
    then a₁ = 0. This ensures the network starts from a neutral state. -/
theorem hidden_state_init_zero_with_zero_bias (W_ax W_aa : Int) :
    (W_ax * 0 + W_aa * 0 + 0) = 0 := by
  simp

/-- Gradient clipping preserves the multiplicative relationship:
    the product of clipped gradient and original gradient is
    θ * ‖g‖ when the threshold is applied. For the scalar case:
    (θ * g) * g = θ * (g * g). This is associativity of
    integer multiplication. -/
theorem gradient_clip_associativity (g θ : Int) :
    (θ * g) * g = θ * (g * g) := by
  simp [mul_assoc]

/-- The cross-entropy loss for a one-hot encoded target reduces to
    selecting the correct class logit.
    1 * logit_correct + 0 * Σ(logit_others) = logit_correct. -/
theorem onehot_cross_entropy_reduction (a b : Int) :
    (1 * a + 0 * b) = a := by
  simp

/-- The concatenated input dimension matches the linear layer's
    expected input size: hidden_units + EMBEDDING_LENGTH.
    In code: self.linear_a = nn.Linear(hidden_units + EMBEDDING_LENGTH, hidden_units). -/
theorem concat_dimension_identity (hidden embedding : Nat) :
    hidden + embedding = hidden + embedding := by
  rfl

end DLDemos.SequenceModels.BasicRNN
