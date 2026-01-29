# Certified Proof-Carrying Circuits

> **Status: Work In Progress** — Core pipeline exists but has critical incomplete components.

## Overview

The **Certified Proof-Carrying Circuits** system bridges mechanistic interpretability and formal verification to provide certified guarantees about neural network behavior. This pipeline extracts simplified "circuits" from trained models, computes certified error bounds, and formally verifies safety and correctness properties.

### Target: Martian Interpretability Challenge

This approach targets the [Martian Interpretability Challenge](https://withmartian.com/prize) by providing:

- **Mechanistic** proofs (not correlational analysis)
- **Ground truth** verification against MBPP-Lean specifications
- **Scalable** extraction from 1.3B to 70B parameter models
- **Generalizable** results across multiple code LLMs

---

## Implementation Status

### Component Overview

| Component | Status | Blocker | Location |
|-----------|--------|---------|----------|
| **A: Circuit Extraction** | ⚠️ 70% | `_evaluate_circuit()` stub | `extraction/` |
| **B: Translation Layer** | ✅ 85% | Minor issues | `translator/circuit_to_lean.py` |
| **C: Lean Verification** | ❌ 40% | 16 `sorry` placeholders | `lean/FormalVerifML/` |

### Component A: BlockCert Extraction

**Location:** `extraction/circuit_extractor.py`

| Feature | Status | Notes |
|---------|--------|-------|
| `CircuitExtractor` class | ✅ Implemented | |
| `compute_importance_scores()` | ✅ Implemented | Gradient and activation metrics |
| `activation_patching()` | ⚠️ Partial | Not fully integrated |
| `edge_pruning()` | ✅ Implemented | |
| `extract_circuit()` | ✅ Implemented | |
| `compute_lipschitz_constant()` | ✅ Implemented | Linear, ReLU, LayerNorm |
| `compute_error_bound()` | ⚠️ Partial | Depends on stub below |
| `_evaluate_circuit()` | ❌ **STUB** | **Critical blocker** |
| `export_to_json()` | ✅ Implemented | With SHA-256 hash |

**Critical Issue:**

```python
# extraction/circuit_extractor.py:340-353
def _evaluate_circuit(self, circuit_components, inputs):
    """
    Evaluate the extracted circuit (simplified implementation)

    In practice, this would construct a new model with only the
    circuit components and evaluate it.
    """
    # Simplified: return original model output
    # A full implementation would build a sparse model
    return self.model(inputs)  # ← RETURNS WRONG OUTPUT
```

This stub returns the original model's output instead of building and evaluating the sparse circuit. **All error bounds computed using this are inaccurate.**

### Component B: Translation Layer

**Location:** `translator/circuit_to_lean.py`

| Feature | Status | Notes |
|---------|--------|-------|
| `CircuitToLeanTranslator` class | ✅ Implemented | |
| Sparse weight formatting | ✅ Implemented | Edge-list representation |
| Error bound definitions | ✅ Implemented | |
| CLI interface | ✅ Implemented | |
| Batch translation | ✅ Implemented | |

**Output Example:**
```lean
-- Sparse representation (tractable)
def component_0_edges : List CircuitEdge := [
  ⟨0, 0, 0.5⟩,  -- source=0, target=0, weight=0.5
  ⟨3, 0, 0.3⟩,  -- source=3, target=0, weight=0.3
  ⟨2, 1, 0.2⟩   -- source=2, target=1, weight=0.2
]
```

### Component C: Lean 4 Verification

**Location:** `lean/FormalVerifML/`

#### Definitions (Complete)

| Structure | Status | Location |
|-----------|--------|----------|
| `Circuit` | ✅ Complete | `base/circuit_models.lean` |
| `CircuitComponent` | ✅ Complete | `base/circuit_models.lean` |
| `CircuitEdge` | ✅ Complete | `base/circuit_models.lean` |
| `ErrorBound` | ✅ Complete | `base/circuit_models.lean` |
| `evalCircuit` | ✅ Complete | `base/circuit_models.lean` |
| `applySparseLinear` | ✅ Complete | `base/circuit_models.lean` |
| `circuitRobust` | ✅ Complete | `base/circuit_models.lean` |
| `circuitMonotonic` | ✅ Complete | `base/circuit_models.lean` |

#### Theorems (Incomplete)

| Theorem | Priority | Status | Location |
|---------|----------|--------|----------|
| `property_transfer` | **P0** | ❌ `sorry` | `base/circuit_models.lean:217` |
| `lipschitz_composition_bound` | **P0** | ❌ `sorry` | `base/circuit_models.lean:203` |
| `circuit_robustness_example` | P1 | ❌ `sorry` | `proofs/circuit_proofs.lean:48` |
| `circuit_monotonic_example` | P1 | ❌ `sorry` | `proofs/circuit_proofs.lean:91` |
| `complete_circuit_verification` | P1 | ❌ `sorry` | `proofs/circuit_proofs.lean:248` |
| 11 other theorems | P2-P3 | ❌ `sorry` | Various |

**Total: 16 theorems with `sorry` placeholders**

See [PROOF_ROADMAP.md](PROOF_ROADMAP.md) for complete list and priorities.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CERTIFIED CIRCUITS PIPELINE               │
│                     (Work in Progress)                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Component A    │      │  Component B    │      │  Component C    │
│                 │      │                 │      │                 │
│  BlockCert      │ ───> │  Translation    │ ───> │  Lean 4         │
│  Extraction     │      │  Layer          │      │  Verification   │
│                 │      │                 │      │                 │
│  ⚠️ 70%         │      │  ✅ 85%         │      │  ❌ 40%         │
│  (stub blocks)  │      │  (working)      │      │  (sorry blocks) │
└─────────────────┘      └─────────────────┘      └─────────────────┘
        │                         │                         │
        v                         v                         v
  circuit.json            circuit.lean              proofs.lean
  + certificate           + definitions             + theorems
  (⚠️ inaccurate)         (✅ valid)                (❌ incomplete)
```

---

## Key Algorithm: Lipschitz Composition

**Purpose:** Compute certified error bounds for circuit approximation.

**Theory:**
- Let `B_i` be the original block and `B̂_i` be the surrogate circuit
- Let `ε_i` be the local error: `‖B̂_i(x) - B_i(x)‖ ≤ ε_i`
- Let `L_i` be the Lipschitz constant of block `i`
- Global error bound: `‖F̂(x) - F(x)‖ ≤ Σ_i (ε_i ∏_{j>i} L_j)`

**Implementation Status:**
- Python computation: ⚠️ Implemented but uses stub for circuit evaluation
- Lean theorem: ❌ `lipschitz_composition_bound` has `sorry`

**Critical Risk: Bound Explosion**

Error bounds can compound across layers and become vacuous (useless).

**Validation Required:**
```python
ratio = theoretical_bound / empirical_max_error
if ratio > 100:
    # Bounds are too loose - proofs will be vacuous
    raise Error("Lipschitz bounds exploded")
```

---

## Usage (Current State)

### End-to-End Demo

```bash
# Runs on toy model - demonstrates pipeline but uses stubs
python examples/end_to_end_pipeline.py
```

### Step-by-Step (with Caveats)

#### Step 1: Extract Circuit

```python
from extraction.circuit_extractor import extract_transformer_circuit

# WARNING: Error bounds are inaccurate due to _evaluate_circuit() stub
circuit_data = extract_transformer_circuit(
    model=your_pytorch_model,
    calibration_data=calibration_inputs,
    calibration_targets=calibration_outputs,
    test_data=test_inputs,
    test_targets=test_outputs,
    output_path="circuit.json",
    pruning_threshold=0.01
)
```

#### Step 2: Translate to Lean

```bash
python translator/circuit_to_lean.py \
    --circuit_json circuit.json \
    --output_dir lean/FormalVerifML/generated
```

#### Step 3: Attempt Verification

```bash
# Will compile but proofs are incomplete (sorry)
lake build
```

---

## What Works Today

1. ✅ Circuit extraction identifies important components
2. ✅ Sparse edge representation is generated correctly
3. ✅ Lean code is syntactically valid and type-checks
4. ✅ Lean definitions (structures, functions) are complete
5. ✅ JSON export includes certificate hash

## What Does NOT Work Today

1. ❌ `_evaluate_circuit()` returns wrong output (stub)
2. ❌ Error bounds are inaccurate without proper circuit evaluation
3. ❌ Core theorems have `sorry` - no actual proofs
4. ❌ MBPP-Lean benchmark integration not implemented
5. ❌ Cross-model comparison not implemented

---

## Required Work

### Critical Path (Must Complete First)

| Task | Expert Needed | Effort |
|------|---------------|--------|
| Fix `_evaluate_circuit()` | MI/PyTorch engineer | 2-3 days |
| Validate Lipschitz tightness | MI engineer | 1-2 days |
| Complete `lipschitz_composition_bound` | Lean expert | 1 week |
| Complete `property_transfer` | Lean expert | 1 week |

### MBPP Integration (Phase 2)

| Task | Expert Needed | Effort |
|------|---------------|--------|
| Implement `fetch_dataset.py` | Python engineer | 1 day |
| Implement `run_benchmark.py` | Python engineer | 2-3 days |
| Test on code LLMs | ML engineer | 1 week |

---

## Theoretical Foundation

Based on:
1. **BlockCert** - Certified interpretability via Lipschitz composition
2. **Mechanistic Interpretability** - Transformer circuits research (Anthropic)
3. **Formal Verification** - Lean 4 theorem prover

### Key Insight: Property Transfer

If we can prove:
1. Property P holds on sparse circuit F̂
2. Circuit approximates model: `‖F̂(x) - F(x)‖ < ε`
3. Property P is Lipschitz with constant L_P

Then: Model F satisfies P within `L_P * ε`.

**This is the core theorem (`property_transfer`) that currently has `sorry`.**

---

## References

1. **BlockCert**: Certified Approach to Mechanistic Interpretability
2. **Lean 4**: [The Lean Theorem Prover](https://leanprover.github.io/)
3. **Transformer Circuits**: [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/)
4. **VERINA**: [MBPP-Lean Benchmark](https://github.com/sunblaze-ucb/verina)
5. **Martian Challenge**: [Interpretability Prize](https://withmartian.com/prize)

---

## Contributing

We need:
- **Lean experts** to complete the `sorry` theorems
- **MI researchers** to fix circuit evaluation and validate bounds
- **ML engineers** to run experiments on code LLMs

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

**Last Updated**: January 2026
**Status**: Work in Progress
**Target**: Martian Interpretability Challenge
