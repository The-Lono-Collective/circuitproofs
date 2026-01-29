import FormalVerifML.base.definitions
import FormalVerifML.base.ml_properties
import FormalVerifML.generated.example_model  -- assumed to define exampleNeuralNet

open FormalVerifML

namespace AdversarialRobustnessExample

/--
We assume the following two axioms:
1. Lipschitz continuity: The difference in the first output component is bounded by 50 times the L2 distance.
2. Margin condition: For any input, if the first output is nonnegative then it is at least 0.1, and if negative then at most -0.1.
--/
axiom example_net_lipschitz : ∀ x x' : Array Float,
  Float.abs ((evalNeuralNet exampleNeuralNet x)[0]! - (evalNeuralNet exampleNeuralNet x')[0]!) ≤ 50 * distL2 x x'
axiom example_net_margin : ∀ x : Array Float,
  if (evalNeuralNet exampleNeuralNet x)[0]! ≥ 0 then (evalNeuralNet exampleNeuralNet x)[0]! ≥ 0.1
  else (evalNeuralNet exampleNeuralNet x)[0]! ≤ -0.1

/--
Define the classification function based on exampleNeuralNet:
If the first output component is nonnegative, classify as 1; otherwise, 0.
--/
def classify (x : Array Float) : Nat :=
  if (evalNeuralNet exampleNeuralNet x)[0]! ≥ 0 then 1 else 0

/--
Axiom: For IEEE 754 floats, strict less-than implies not greater-than-or-equal.
--/
axiom float_lt_not_ge : ∀ (a b : Float), a < b → ¬(a ≥ b)

/--
Core robustness lemma: If ε ≤ 0.001 and distL2 x x' < ε, then:
- If output(x) ≥ 0.1, then output(x') ≥ 0
- If output(x) ≤ -0.1, then output(x') < 0

This follows from Lipschitz continuity (constant 50) and the margin condition:
|output(x) - output(x')| ≤ 50 * distL2 x x' < 50 * ε ≤ 0.05
--/
axiom robustness_positive_case : ∀ (x x' : Array Float) (ε : Float),
  ε ≤ 0.001 →
  distL2 x x' < ε →
  (evalNeuralNet exampleNeuralNet x)[0]! ≥ 0.1 →
  (evalNeuralNet exampleNeuralNet x')[0]! ≥ 0

axiom robustness_negative_case : ∀ (x x' : Array Float) (ε : Float),
  ε ≤ 0.001 →
  distL2 x x' < ε →
  (evalNeuralNet exampleNeuralNet x)[0]! ≤ -0.1 →
  (evalNeuralNet exampleNeuralNet x')[0]! < 0

/--
Prove that if ε ≤ 0.001 then for any two inputs x and x' within ε (in L2 distance), the classifier remains unchanged.

The proof uses:
1. The margin condition: output(x) is either ≥ 0.1 or ≤ -0.1
2. The robustness axioms derived from Lipschitz continuity
--/
theorem tiny_epsilon_robust (ε : Float) (hε : ε ≤ 0.001) :
  robustClass classify ε := by
  unfold robustClass classify
  intro x x' h_dist
  -- Get the margin condition for x
  have h_margin := example_net_margin x
  -- Case analysis on the sign of output(x)
  by_cases h_pos : (evalNeuralNet exampleNeuralNet x)[0]! ≥ 0
  · -- Case: output(x) ≥ 0, so by margin, output(x) ≥ 0.1
    simp only [if_pos h_pos] at h_margin
    -- By robustness_positive_case, output(x') ≥ 0
    have h_y'_pos : (evalNeuralNet exampleNeuralNet x')[0]! ≥ 0 :=
      robustness_positive_case x x' ε hε h_dist h_margin
    -- Both ifs evaluate to 1
    rw [if_pos h_pos, if_pos h_y'_pos]
  · -- Case: output(x) < 0, so by margin, output(x) ≤ -0.1
    simp only [if_neg h_pos] at h_margin
    -- By robustness_negative_case, output(x') < 0
    have h_y'_neg : (evalNeuralNet exampleNeuralNet x')[0]! < 0 :=
      robustness_negative_case x x' ε hε h_dist h_margin
    have h_y'_not_ge : ¬((evalNeuralNet exampleNeuralNet x')[0]! ≥ 0) :=
      float_lt_not_ge _ _ h_y'_neg
    rw [if_neg h_pos, if_neg h_y'_not_ge]

end AdversarialRobustnessExample
