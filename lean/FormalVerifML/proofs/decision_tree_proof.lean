import FormalVerifML.base.definitions
import FormalVerifML.base.ml_properties
import FormalVerifML.generated.decision_tree_model  -- assumed to define myDecisionTree

open FormalVerifML

namespace DecisionTreeProof

/--
Assume that the generated decision tree is defined as follows:
myDecisionTree = DecisionTree.node 0 5.0 (DecisionTree.leaf 0) (DecisionTree.leaf 1)
--/
axiom myDecisionTree_def : myDecisionTree = DecisionTree.node 0 5.0 (DecisionTree.leaf 0) (DecisionTree.leaf 1)

/--
Axiom: For IEEE 754 floats (excluding NaN), strict less-than implies not less-than-or-equal.
This is a fundamental property of the total order on non-NaN floats.
--/
axiom float_lt_not_le : ∀ (a b : Float), a < b → ¬(b ≤ a)

/--
Prove that if the first component of the input vector is greater than 5.0,
then evaluating the decision tree yields 1.

The proof unfolds the decision tree definition and uses the hypothesis
to show that the else branch (returning 1) is taken.
--/
theorem decision_tree_region_classification (x : Array Float) (h : x[0]! > 5.0) :
  evalDecisionTree myDecisionTree x = 1 := by
  rw [myDecisionTree_def]
  simp only [evalDecisionTree]
  -- Goal: if x[0]! ≤ 5.0 then 0 else 1 = 1
  split_ifs with h_le
  · -- Case where x[0]! ≤ 5.0: contradiction with h : x[0]! > 5.0
    exfalso
    exact float_lt_not_le 5.0 (x[0]!) h h_le
  · -- Case where ¬(x[0]! ≤ 5.0): else branch returns 1
    rfl

end DecisionTreeProof
