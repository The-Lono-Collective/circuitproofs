import FormalVerifML.base.definitions
import FormalVerifML.base.ml_properties
import FormalVerifML.generated.example_model  -- assumed to define exampleNeuralNet

open FormalVerifML

namespace ExtendedRobustness

/--
Assume that each layer is Lipschitz continuous with constant 2.
(This is a realistic assumption if, for example, the weights are normalized.)
-/
axiom layer_lipschitz : ∀ (l : LayerType) (x y : Array Float),
  distL2 (evalLayer l x) (evalLayer l y) ≤ 2 * distL2 x y

/--
Prove that the neural network is Lipschitz continuous with constant 2^(n),
where n is the number of layers.

Note: This theorem uses sorry as a placeholder. A complete proof would use
induction on the number of layers and compose the Lipschitz bounds.
-/
theorem neural_net_lipschitz (nn : NeuralNet) (x x' : Array Float) :
  distL2 (evalNeuralNet nn x) (evalNeuralNet nn x') ≤ (2^(nn.layers.length)) * distL2 x x' := by
  sorry

end ExtendedRobustness
