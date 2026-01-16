import FormalVerifML.base.definitions

namespace FormalVerifML

/-- Example decision tree for testing classification proofs -/
def myDecisionTree : DecisionTree :=
  DecisionTree.node 0 5.0 (DecisionTree.leaf 0) (DecisionTree.leaf 1)

end FormalVerifML
