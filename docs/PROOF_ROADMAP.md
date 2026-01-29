# Proof Completion Roadmap

> **Target:** [Martian Interpretability Challenge](https://withmartian.com/prize)

This document tracks `sorry` placeholders in the Lean codebase and prioritizes proofs for the Martian challenge submission.

---

## Summary

| Priority | Count | Status | Focus |
|----------|-------|--------|-------|
| **P0 (Critical)** | 2 | ❌ Not started | Core value proposition |
| **P1 (High)** | 3 | ❌ Not started | Key verification demos |
| **P2 (Medium)** | 4 | ❌ Not started | Supporting guarantees |
| **P3 (Low)** | 7+ | ❌ Not started | Nice-to-have |

**Total: 16 theorems with `sorry` placeholders**

---

## Priority Levels

- **P0 (Critical)**: Must complete for Martian submission. Without these, the approach has no value.
- **P1 (High)**: Strongly needed for compelling demo. Shows the system works.
- **P2 (Medium)**: Strengthens the submission. Demonstrates breadth.
- **P3 (Low)**: Can defer. Complete if time permits.

---

## P0: Critical Path (MUST COMPLETE)

These two theorems are the **core value proposition**. Without them, the entire approach is unsubstantiated.

### 1. Property Transfer Theorem

**File:** `lean/FormalVerifML/base/circuit_models.lean:217`

```lean
theorem property_transfer (circuit : Circuit)
    (originalModel : Array Float → Array Float)
    (property : Array Float → Prop)
    (propertyLipschitz : Float) :
  circuitSatisfiesProperty circuit property 1.0 →
  circuitApproximatesModel circuit originalModel →
  circuit.errorBound.epsilon < propertyLipschitz →
  (∀ x, property (originalModel x))
```

**Why Critical:** This is THE theorem. It proves that properties verified on the circuit transfer to the original model. Without this, we cannot claim that circuit verification implies model verification.

**Proof Strategy:**
1. Use triangle inequality on the property
2. Apply Lipschitz bound of property
3. Combine with circuit error bound
4. Show margin is sufficient

**Estimated Effort:** 1 week (Lean expert)

### 2. Lipschitz Composition Bound

**File:** `lean/FormalVerifML/base/circuit_models.lean:203`

```lean
theorem lipschitz_composition_bound (circuit : Circuit) :
  -- Error bound is valid via Lipschitz composition
  ...
```

**Why Critical:** Justifies the BlockCert-style error bound computation. Without this, error bounds are unjustified.

**Proof Strategy:**
1. Induction over circuit components
2. Apply Lipschitz composition at each layer
3. Sum local errors with Lipschitz products

**Estimated Effort:** 1 week (Lean expert)

---

## P1: High Priority (Strongly Needed)

These complete the demo and show the system works on concrete examples.

### 3. Circuit Robustness Example

**File:** `lean/FormalVerifML/proofs/circuit_proofs.lean:48`

```lean
theorem circuit_robustness_example (circuit : Circuit) (δ ε : Float) :
  -- Concrete robustness proof for example circuit
  ...
```

**Why Important:** Robustness is the most common property people want to verify. A working example is compelling.

**Estimated Effort:** 3-4 days

### 4. Circuit Monotonicity Example

**File:** `lean/FormalVerifML/proofs/circuit_proofs.lean:91`

```lean
theorem circuit_monotonic_example (circuit : Circuit) (featureIdx : Nat) :
  -- Prove monotonicity for a specific feature
  ...
```

**Why Important:** Monotonicity is key for fairness (e.g., "higher score should not decrease approval"). Good for policy relevance.

**Estimated Effort:** 3-4 days

### 5. Complete Circuit Verification

**File:** `lean/FormalVerifML/proofs/circuit_proofs.lean:248`

```lean
theorem complete_circuit_verification :
  -- Full verification workflow combining all steps
  ...
```

**Why Important:** Showcase theorem combining extraction → translation → verification → certificate.

**Estimated Effort:** 2-3 days

---

## P2: Medium Priority (Strengthens Submission)

### 6. Structural Implies Semantic Equivalence

**File:** `lean/FormalVerifML/base/circuit_equivalence.lean:42`

```lean
theorem structural_implies_semantic ...
```

**Why Important:** Enables counterfactual testing—proving circuits from different inputs are equivalent.

### 7. Fairness Proof Example

**File:** `lean/FormalVerifML/proofs/example_fairness_proof.lean:53`

```lean
theorem example_demographic_parity ...
```

**Why Important:** Fairness is key for regulatory applications. Martian mentions policy relevance.

### 8. Error Bound Validity

**File:** `lean/FormalVerifML/proofs/circuit_proofs.lean:138`

```lean
theorem error_bound_valid (circuit : Circuit) : ...
```

### 9. Structural Equivalence Symmetry/Reflexivity

**File:** `lean/FormalVerifML/base/circuit_equivalence.lean:47-53`

---

## P3: Low Priority (Defer If Needed)

| Theorem | File | Notes |
|---------|------|-------|
| `simpleLinearCircuit_sparse` | `circuit_proofs.lean:32` | Sparsity computation |
| Decision tree verification | `decision_tree_proof.lean:26` | Not code LLM focused |
| Extended robustness | `extended_robustness_proof.lean:25` | Extension of P1 |
| Interpretability proofs | `circuit_proofs.lean:222` | Nice to have |
| 3+ additional theorems | Various | Lower priority |

---

## Recommended Approach

### Week 1: P0 Theorems

1. **Start with `lipschitz_composition_bound`**
   - More foundational
   - `property_transfer` likely depends on it

2. **Then `property_transfer`**
   - Uses Lipschitz bound
   - Core deliverable

### Week 2: P1 Theorems

3. **`circuit_robustness_example`**
   - Concrete demo
   - Uses P0 theorems

4. **`circuit_monotonic_example`**
   - Fairness angle
   - Policy relevance

5. **`complete_circuit_verification`**
   - Showcase theorem
   - Combines everything

### Week 3+: P2 If Time Permits

6. Counterfactual testing support
7. Fairness example
8. Additional properties

---

## Technical Notes

### Working with Lean 4

```bash
# Build and check for errors
lake build

# Check specific file
lake build FormalVerifML.proofs.circuit_proofs

# Interactive development
# Use VS Code with lean4 extension
```

### Useful Tactics

```lean
-- For array/list proofs
simp [List.map, Array.get]

-- For numeric bounds
linarith
omega

-- For function application
apply
exact

-- For structure destruction
cases h
obtain ⟨a, b, c⟩ := h
```

### Key Lemmas to Use

From mathlib:
- `Real.dist_triangle` - Triangle inequality
- `Lipschitz.dist_le_mul` - Lipschitz bound application
- `List.sum_le_sum` - Sum inequalities

From our codebase:
- `evalCircuit_bounded` - Circuit output bounds
- `applySparseLinear_lipschitz` - Sparse layer Lipschitz

---

## Dependencies

```
property_transfer
    ├── lipschitz_composition_bound
    ├── circuitApproximatesModel (definition)
    └── circuitSatisfiesProperty (definition)

circuit_robustness_example
    ├── property_transfer
    └── circuitRobust (definition)

complete_circuit_verification
    ├── property_transfer
    ├── circuit_robustness_example
    └── lipschitz_composition_bound
```

---

## Getting Help

### For Lean Questions

- [Lean 4 Documentation](https://leanprover.github.io/lean4/doc/)
- [Mathlib4 Docs](https://leanprover-community.github.io/mathlib4_docs/)
- [Lean Zulip Chat](https://leanprover.zulipchat.com/)

### For Project Context

- See `docs/CERTIFIED_CIRCUITS.md` for theoretical background
- See `ROADMAP.md` for project priorities
- See `extraction/circuit_extractor.py` for Python implementation

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-20 | Reprioritized for Martian challenge |
| 2026-01-20 | Added detailed proof strategies |
| 2026-01-16 | Initial roadmap |
