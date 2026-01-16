import FormalVerifML.base.definitions

namespace FormalVerifML

/-- Example neural network for testing robustness proofs -/
def exampleNeuralNet : NeuralNet :=
  { inputDim := 2,
    outputDim := 1,
    layers := [
      LayerType.linear #[#[0.5, -0.3], #[0.2, 0.8]] #[0.1, -0.1],
      LayerType.relu,
      LayerType.linear #[#[0.7, -0.4]] #[0.05]
    ] }

end FormalVerifML
