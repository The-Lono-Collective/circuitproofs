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

**Location:** `extraction/blockcert/`

| Module | Status | Notes |
|---------|--------|-------|
| `ir.py` (BlockIR, TraceRecord, TraceDataset) | ✅ Implemented | Intermediate representation and trace data |
| `interpreter.py` (BlockInterpreter) | ✅ Implemented | Block evaluation |
| `certifier.py` (BlockCertifier) | ✅ Implemented | Certification with Lipschitz bounds |
| `certificate.py` (Certificate) | ✅ Implemented | Certificate generation |
| SheafCert Pipeline (CD-T + DiscoGP) | ❌ **Not implemented** | Planned replacement for archived `circuit_extractor.py` |

**Note:** The legacy `circuit_extractor.py` has been archived to the `archive/legacy-blockcert` branch. The SheafCert extraction pipeline is planned but not yet implemented.

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

### Step-by-Step

#### Step 1: Extract Circuit

```python
from extraction.blockcert import BlockCertifier, BlockIR, BlockInterpreter, Certificate

# BlockCert modules provide IR, interpretation, certification, and certificate generation.
# The SheafCert extraction pipeline (CD-T + DiscoGP) is planned but not yet implemented.
# See extraction/blockcert/ for available modules.
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

1. ✅ BlockCert IR, interpreter, certifier, and certificate modules implemented
2. ✅ Sparse edge representation is generated correctly
3. ✅ Lean code is syntactically valid and type-checks
4. ✅ Lean definitions (structures, functions) are complete
5. ✅ JSON export includes certificate hash

## What Does NOT Work Today

1. ❌ SheafCert extraction pipeline not yet implemented (legacy `circuit_extractor.py` archived)
2. ❌ Core theorems have `sorry` - no actual proofs
3. ❌ MBPP-Lean benchmark integration not implemented
4. ❌ Cross-model comparison not implemented

---

## Required Work

### Critical Path (Must Complete First)

| Task | Expert Needed | Effort |
|------|---------------|--------|
| Implement SheafCert pipeline | MI/PyTorch engineer | TBD |
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
