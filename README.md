# CircuitProofs: Formal Verification of Neural Network Circuits

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Lean 4](https://img.shields.io/badge/Lean-4-green.svg)](https://leanprover.github.io/)

> **Prove that extracted circuits match formal specifications** — not correlation, but mathematical certainty.

This project targets the [Martian Interpretability Challenge](https://withmartian.com/prize), addressing the core problem: current interpretability is correlational, not mechanistic. We provide **formal proofs** that extracted circuits implement specific algorithms.

---

## Project Status: Work In Progress

| Component | Status | Description |
|-----------|--------|-------------|
| **Circuit Extraction** | ⚠️ 70% | Core works, `_evaluate_circuit()` is stub |
| **Circuit → Lean Translation** | ✅ 85% | Generates valid Lean code |
| **Lean Core Definitions** | ✅ 100% | All structures defined |
| **Lean Proofs** | ❌ 40% | 16 theorems have `sorry` placeholders |
| **MBPP-Lean Benchmark** | ❌ 10% | Scaffolding only, not implemented |
| **Distributed Extraction** | ⚠️ 60% | Infrastructure exists, not tested at scale |

**Critical Blockers:**
- `_evaluate_circuit()` stub prevents accurate error bounds
- `property_transfer` theorem incomplete (core value proposition)
- `lipschitz_composition_bound` theorem incomplete
- MBPP benchmark runner not implemented

---

## The Approach

### Why This Wins

Most interpretability work says: *"This circuit **seems** to compute X based on our analysis."*

We say: *"This circuit is **proven** to compute X. Here's the Lean proof."*

### Architecture

```
Code LLM solves MBPP problem (e.g., "sort a list")
         ↓
Extract sparse circuit from model (Component A)
         ↓
Translate circuit to Lean 4 (Component B)
         ↓
Prove circuit matches MBPP Lean specification (Component C)
         ↓
Certificate: "Circuit implements sorting algorithm within ε error"
```

### Key Innovation: Ground Truth Verification

The [MBPP-Lean benchmark](benchmarks/verina/) provides 49 programming problems with **pre-written Lean specifications**. These are the ground truth. We:

1. Have a code LLM solve the problem
2. Extract the circuit it uses
3. **Prove** the circuit matches the specification

This is mechanistic (causal), not correlational (pattern-matching).

---

## Project Scope

### In Scope (Martian Challenge Focus)

| Feature | Purpose |
|---------|---------|
| Circuit extraction from code LLMs | Identify computational subgraphs |
| Lipschitz error bounds | Certify approximation quality |
| Lean 4 formal verification | Mathematical proofs |
| MBPP-Lean benchmark | Ground truth specifications |
| Cross-model generalization | Same algorithm → same circuit |
| Distributed extraction | Scale to 70B parameter models |

### Out of Scope (Deprioritized)

| Feature | Status | Reason |
|---------|--------|--------|
| Vision models (ViT, CLIP) | Deprioritized | Not code generation |
| Enterprise features (auth, audit) | Deprioritized | Not needed for challenge |
| Web interface | Deprioritized | CLI sufficient |
| Generic HuggingFace support | Deprioritized | Focus on code LLMs only |

---

## Target Models

| Model | Size | Purpose |
|-------|------|---------|
| DeepSeek-Coder-1.3B | 1.3B | Fast iteration |
| StarCoder-7B | 7B | Main experiments |
| CodeLlama-34B | 34B | Generalization |
| CodeLlama-70B | 70B | Scale demonstration |

---

## Quick Start

### Prerequisites

- Python 3.9+
- Lean 4 (v4.18.0-rc1)
- PyTorch 2.0+
- 8GB+ RAM (more for larger models)

### Installation

```bash
git clone https://github.com/the-lono-collective/circuitproofs.git
cd circuitproofs

# Python dependencies
pip install -r translator/requirements.txt

# Build Lean project
lake build
```

### Run End-to-End Demo (Current State)

```bash
# This demonstrates the pipeline on a toy model
# Note: Uses stubs for circuit evaluation
python examples/end_to_end_pipeline.py
```

### Run Circuit Extraction (Partial)

```python
from extraction.circuit_extractor import extract_transformer_circuit

# Note: _evaluate_circuit() is a stub - error bounds are approximate
circuit_data = extract_transformer_circuit(
    model=your_model,
    calibration_data=calib_data,
    calibration_targets=calib_targets,
    test_data=test_data,
    test_targets=test_targets,
    output_path="circuit.json",
    pruning_threshold=0.01
)
```

---

## Implementation Status Details

### Component A: Circuit Extraction (`extraction/`)

| Feature | Status | Notes |
|---------|--------|-------|
| `CircuitExtractor` class | ✅ Implemented | |
| Gradient-based importance | ✅ Implemented | |
| Activation-based importance | ✅ Implemented | |
| Edge pruning | ✅ Implemented | |
| Lipschitz constant estimation | ✅ Implemented | |
| `_evaluate_circuit()` | ❌ **STUB** | Returns original model output |
| Error bound computation | ⚠️ Partial | Depends on stub above |
| JSON export with hash | ✅ Implemented | |

**Critical Issue:** `_evaluate_circuit()` at `extraction/circuit_extractor.py:340` is a stub. It returns the original model's output instead of evaluating the sparse circuit. This means error bounds are not accurate.

### Component B: Translation (`translator/`)

| Feature | Status | Notes |
|---------|--------|-------|
| `CircuitToLeanTranslator` | ✅ Implemented | |
| Sparse weight formatting | ✅ Implemented | |
| Error bound definitions | ✅ Implemented | |
| CLI interface | ✅ Implemented | |
| Batch translation | ✅ Implemented | |

### Component C: Lean Verification (`lean/FormalVerifML/`)

| Feature | Status | Notes |
|---------|--------|-------|
| Core structures | ✅ Complete | `Circuit`, `CircuitComponent`, `ErrorBound` |
| Evaluation functions | ✅ Complete | `evalCircuit`, `applySparseLinear` |
| Property definitions | ✅ Complete | `circuitRobust`, `circuitMonotonic` |
| `property_transfer` theorem | ❌ **SORRY** | Core value proposition |
| `lipschitz_composition_bound` | ❌ **SORRY** | Error bound justification |
| Circuit robustness proof | ❌ **SORRY** | |
| 13 other theorems | ❌ **SORRY** | See `docs/PROOF_ROADMAP.md` |

### MBPP-Lean Benchmark (`benchmarks/verina/`)

| Feature | Status | Notes |
|---------|--------|-------|
| README documentation | ✅ Complete | |
| `fetch_dataset.py` | ❌ Not implemented | Scaffold only |
| `run_benchmark.py` | ❌ Not implemented | Scaffold only |
| `variant_generator.py` | ❌ Not implemented | For counterfactual testing |
| `circuit_comparator.py` | ❌ Not implemented | For cross-model comparison |

---

## Development Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed phases and timeline.

### Phase 1: Critical Path (Weeks 1-2)
- [ ] Fix `_evaluate_circuit()` stub
- [ ] Validate Lipschitz bound tightness (< 100x empirical)
- [ ] Complete `property_transfer` theorem
- [ ] Complete `lipschitz_composition_bound` theorem

### Phase 2: MBPP Integration (Weeks 3-4)
- [ ] Implement `fetch_dataset.py`
- [ ] Implement `run_benchmark.py`
- [ ] Test on DeepSeek-Coder-1.3B with 10 MBPP problems

### Phase 3: Scale & Generalize (Weeks 5-6)
- [ ] Run on StarCoder-7B, CodeLlama-34B
- [ ] Demonstrate cross-model circuit similarity
- [ ] Scale to CodeLlama-70B

### Phase 4: Submission (Weeks 7-8)
- [ ] Write results analysis
- [ ] Prepare Martian challenge submission

---

## Known Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Lipschitz bound explosion** | Critical | Tightness gate: ratio < 100x |
| Proofs harder than expected | High | Start with simplest theorems |
| Circuits don't recover algorithms | High | Start with simple MBPP problems |
| 70B extraction OOMs | Medium | Distributed infrastructure |

---

## Repository Structure

```
circuitproofs/
├── extraction/                 # Component A: Circuit extraction
│   ├── circuit_extractor.py   # Main extraction (⚠️ has stub)
│   └── example_extraction.py  # Demo script
├── translator/                 # Component B: Translation
│   ├── circuit_to_lean.py     # Circuit → Lean
│   └── generate_lean_model.py # Generic model → Lean
├── lean/FormalVerifML/        # Component C: Verification
│   ├── base/                  # Core definitions (✅)
│   │   ├── circuit_models.lean
│   │   └── definitions.lean
│   ├── proofs/                # Theorems (❌ many sorry)
│   │   └── circuit_proofs.lean
│   └── generated/             # Auto-generated models
├── benchmarks/verina/         # MBPP-Lean (❌ not implemented)
├── examples/                  # Demo scripts
├── docs/                      # Documentation
└── webapp/                    # Web UI (deprioritized)
```

---

## Contributing

We need help with:

1. **Lean Expert**: Complete the `sorry` theorems in `proofs/`
2. **MI Researcher**: Fix `_evaluate_circuit()` and validate bounds
3. **ML Engineer**: Implement MBPP benchmark runner
4. **Distributed Systems**: Scale extraction to 70B models

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Attribution

- Extended from [FormalVerifML](https://github.com/fraware/formal_verif_ml)
- MBPP-Lean specifications from [VERINA](https://github.com/sunblaze-ucb/verina)
- Targeting [Martian Interpretability Challenge](https://withmartian.com/prize)

---

## License

MIT License - see [LICENSE](LICENSE)
