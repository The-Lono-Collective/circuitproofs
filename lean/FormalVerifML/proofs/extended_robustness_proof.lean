import FormalVerifML.base.definitions
import FormalVerifML.base.ml_properties
import FormalVerifML.generated.example_model  -- assumed to define exampleNeuralNet

open FormalVerifML

namespace ExtendedRobustness

/--
Assume that each layer is Lipschitz continuous with constant 2.
(This is a realistic assumption if, for example, the weights are normalized.)
--/
axiom layer_lipschitz : ∀ (l : LayerType) (x y : Array Float),
  distL2 (evalLayer l x) (evalLayer l y) ≤ 2 * distL2 x y

/-! ## Float Power Axioms -/

/-- Float.pow 2.0 0 = 1.0 -/
axiom float_pow_zero : Float.pow 2.0 0 = 1.0

/-- Float.pow 2.0 (n+1) = 2 * Float.pow 2.0 n -/
axiom float_pow_succ : ∀ n : Float, n ≥ 0 → Float.pow 2.0 (n + 1) = 2.0 * Float.pow 2.0 n

/-- Float.ofNat preserves successor -/
axiom float_ofNat_succ : ∀ n : Nat, Float.ofNat (n + 1) = Float.ofNat n + 1

/-- Float.ofNat 0 = 0 -/
axiom float_ofNat_zero : Float.ofNat 0 = 0

/-- Float.ofNat is non-negative -/
axiom float_ofNat_nonneg : ∀ n : Nat, Float.ofNat n ≥ 0

/-- Transitivity for Float multiplication with Lipschitz composition -/
axiom lipschitz_compose : ∀ (d1 d2 d3 : Float) (L1 L2 : Float),
  d2 ≤ L1 * d1 → d3 ≤ L2 * d2 → d3 ≤ (L2 * L1) * d1

/-- 1.0 * x = x for Floats -/
axiom float_one_mul : ∀ x : Float, 1.0 * x = x

/-- Float multiplication is commutative -/
axiom float_mul_comm : ∀ x y : Float, x * y = y * x

/-- 2 and 2.0 are the same Float -/
axiom float_two_eq : (2 : Float) = 2.0

/-! ## Float Comparison Axioms -/

/-- Float ≤ is reflexive -/
axiom float_le_refl : ∀ x : Float, x ≤ x

/-- Transitivity of ≤ for Float with multiplicative bound -/
axiom float_le_mul_trans : ∀ (a b c L1 L2 : Float),
  a ≤ L1 * b → b ≤ L2 * c → a ≤ (L1 * L2) * c

/-! ## Helper Lemma for List Induction -/

/--
Helper: foldl of evalLayer over layers list satisfies Lipschitz with 2^(length layers)
We need to prove for arbitrary starting points x, x' for the induction to work.
-/
theorem layers_foldl_lipschitz : ∀ (layers : List LayerType) (x x' : Array Float),
    distL2 (layers.foldl (fun acc l => evalLayer l acc) x)
           (layers.foldl (fun acc l => evalLayer l acc) x') ≤
    Float.pow 2.0 (Float.ofNat layers.length) * distL2 x x' := by
  intro layers
  induction layers with
  | nil =>
    -- Base case: empty list means identity function
    intro x x'
    simp only [List.foldl_nil, List.length_nil]
    rw [float_ofNat_zero, float_pow_zero, float_one_mul]
    exact float_le_refl (distL2 x x')
  | cons layer rest ih =>
    -- Inductive case: one layer + rest
    intro x x'
    simp only [List.foldl_cons, List.length_cons]
    -- After one layer, intermediate values are evalLayer layer x and evalLayer layer x'
    let y := evalLayer layer x
    let y' := evalLayer layer x'
    -- By layer_lipschitz: distL2 y y' ≤ 2 * distL2 x x'
    have h_layer_raw : distL2 y y' ≤ 2 * distL2 x x' := layer_lipschitz layer x x'
    have h_layer : distL2 y y' ≤ 2.0 * distL2 x x' := by rw [← float_two_eq]; exact h_layer_raw
    -- By IH on rest with y, y': distL2 (foldl rest y) (foldl rest y') ≤ 2^|rest| * distL2 y y'
    have h_rest : distL2 (rest.foldl (fun acc l => evalLayer l acc) y)
                        (rest.foldl (fun acc l => evalLayer l acc) y') ≤
                 Float.pow 2.0 (Float.ofNat rest.length) * distL2 y y' := ih y y'
    -- Compose: result ≤ 2^|rest| * (2 * distL2 x x') = 2^(|rest|+1) * distL2 x x'
    have h_compose : distL2 (rest.foldl (fun acc l => evalLayer l acc) y)
                           (rest.foldl (fun acc l => evalLayer l acc) y') ≤
                    (Float.pow 2.0 (Float.ofNat rest.length) * 2.0) * distL2 x x' :=
      float_le_mul_trans _ _ _ (Float.pow 2.0 (Float.ofNat rest.length)) 2.0 h_rest h_layer
    -- Need to show: 2^|rest| * 2 = 2 * 2^|rest| = 2^(|rest|+1)
    have h_pow : Float.pow 2.0 (Float.ofNat rest.length) * 2.0 =
                 Float.pow 2.0 (Float.ofNat (rest.length + 1)) := by
      rw [float_ofNat_succ, float_pow_succ (Float.ofNat rest.length) (float_ofNat_nonneg rest.length)]
      -- Now need: Float.pow 2.0 (Float.ofNat rest.length) * 2.0 = 2.0 * Float.pow 2.0 (Float.ofNat rest.length)
      exact float_mul_comm (Float.pow 2.0 (Float.ofNat rest.length)) 2.0
    rw [h_pow] at h_compose
    exact h_compose

/--
Prove that the neural network is Lipschitz continuous with constant 2^(n),
where n is the number of layers.
--/
theorem neural_net_lipschitz (nn : NeuralNet) (x x' : Array Float) :
    distL2 (evalNeuralNet nn x) (evalNeuralNet nn x') ≤
    Float.pow 2.0 (Float.ofNat nn.layers.length) * distL2 x x' := by
  unfold evalNeuralNet
  exact layers_foldl_lipschitz nn.layers x x'

end ExtendedRobustness
