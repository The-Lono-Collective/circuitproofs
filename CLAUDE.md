# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Focus

**Target:** [Martian Interpretability Challenge](https://withmartian.com/prize) ($1M prize pool)

**Goal:** Prove extracted circuits match formal specifications — not correlation, but mathematical certainty.

**Status:** Work in Progress. See [ROADMAP.md](ROADMAP.md) for current state.

---

## Current Priorities

### In Scope (Work on These)

| Priority | Area | Focus |
|----------|------|-------|
| **P0** | `extraction/circuit_extractor.py` | Fix `_evaluate_circuit()` stub |
| **P0** | `lean/FormalVerifML/base/circuit_models.lean` | Complete `property_transfer`, `lipschitz_composition_bound` |
| **P0** | `lean/FormalVerifML/proofs/circuit_proofs.lean` | Complete `sorry` theorems |
| **P0** | `benchmarks/verina/` | Implement MBPP benchmark runner |
| **P1** | Distributed extraction | Adapt for 70B code LLMs |

### Out of Scope (Do NOT Work on These)

| Area | Reason |
|------|--------|
| `webapp/` | Web UI deprioritized |
| Vision models (ViT, CLIP) | Not code generation |
| Enterprise features (auth, audit) | Not needed for challenge |
| `translator/test_huggingface_models.py` | Vision model tests |
| `translator/test_enterprise_features.py` | Enterprise tests |

---

## Critical Blockers

### 1. `_evaluate_circuit()` Stub

**Location:** `extraction/circuit_extractor.py:340-353`

**Problem:** Returns original model output instead of sparse circuit output. Error bounds are meaningless.

**Fix Required:** Build and evaluate masked/sparse network from circuit components.

### 2. Core Lean Theorems

**`property_transfer`** at `lean/FormalVerifML/base/circuit_models.lean:217`
- Proves properties on circuits transfer to original model
- This is the core value proposition

**`lipschitz_composition_bound`** at `lean/FormalVerifML/base/circuit_models.lean:203`
- Justifies error bound computation

### 3. Lipschitz Bound Tightness

**Risk:** Bounds can explode and become vacuous.

**Validation:** `theoretical_bound / empirical_max_error` must be < 100x.

---

## Development Philosophy

### Test-Driven Development (TDD)

**All code changes MUST follow TDD principles:**

1. **Red**: Write a failing test first
2. **Green**: Write minimum code to pass
3. **Refactor**: Clean up while keeping tests passing

**For Python:**
```bash
python -m pytest translator/tests/ -v
```

**For Lean:**
```bash
lake build
```

### Production-Ready Code Standards

1. **Clean Code**: Single responsibility, DRY, KISS
2. **Documentation**: Docstrings for all public functions
3. **Type Safety**: Type hints for Python, explicit annotations for Lean
4. **Error Handling**: Specific exceptions, meaningful messages

---

## Development Commands

### Building and Testing

```bash
# Build Lean code
lake build

# Run Python tests
python -m pytest translator/tests/ -v

# Run end-to-end demo (uses stubs)
python examples/end_to_end_pipeline.py
```

### Circuit Extraction

```bash
# Extract circuit (note: _evaluate_circuit is stub)
python -c "
from extraction.circuit_extractor import CircuitExtractor
# See examples/end_to_end_pipeline.py for usage
"
```

### Translation

```bash
# Translate circuit JSON to Lean
python translator/circuit_to_lean.py \
    --circuit_json circuit.json \
    --output_dir lean/FormalVerifML/generated
```

---

## Key Files

### Must Modify (P0)

| File | Issue | Action |
|------|-------|--------|
| `extraction/circuit_extractor.py:340` | `_evaluate_circuit()` stub | Implement sparse evaluation |
| `lean/FormalVerifML/base/circuit_models.lean:203` | `sorry` | Complete `lipschitz_composition_bound` |
| `lean/FormalVerifML/base/circuit_models.lean:217` | `sorry` | Complete `property_transfer` |
| `lean/FormalVerifML/proofs/circuit_proofs.lean` | Multiple `sorry` | Complete all proofs |

### Must Create (P0)

| File | Purpose |
|------|---------|
| `benchmarks/verina/fetch_dataset.py` | Download MBPP-Lean dataset |
| `benchmarks/verina/run_benchmark.py` | Run extraction + verification |
| `scripts/validate_tightness.py` | Check Lipschitz bound quality |

### Can Ignore

| File/Directory | Reason |
|----------------|--------|
| `webapp/` | Deprioritized |
| `lean/FormalVerifML/base/vision_models.lean` | Not in scope |
| `lean/FormalVerifML/base/enterprise_features.lean` | Not in scope |

---

## Architecture Overview

```
Code LLM solves MBPP problem
         ↓
Component A: Extract circuit (extraction/)
         ↓
Component B: Translate to Lean (translator/)
         ↓
Component C: Prove matches spec (lean/)
         ↓
Certificate: "Circuit implements algorithm within ε"
```

### Component Status

| Component | Status | Location |
|-----------|--------|----------|
| A: Extraction | ⚠️ 70% (stub) | `extraction/circuit_extractor.py` |
| B: Translation | ✅ 85% | `translator/circuit_to_lean.py` |
| C: Verification | ❌ 40% (sorry) | `lean/FormalVerifML/` |
| MBPP Benchmark | ❌ 10% | `benchmarks/verina/` |

---

## Target Models

| Model | Size | Purpose |
|-------|------|---------|
| DeepSeek-Coder-1.3B | 1.3B | Fast iteration |
| StarCoder-7B | 7B | Main experiments |
| CodeLlama-34B | 34B | Generalization |
| CodeLlama-70B | 70B | Scale demonstration |

---

## Lean 4 Notes

### Version

The project uses **Lean 4 v4.18.0-rc1** (specified in `lean-toolchain`). Do not change without testing all proofs.

### Key Structures

```lean
-- Circuit with error bounds
structure Circuit where
  components : List CircuitComponent
  errorBound : ErrorBound
  certificateHash : String

-- Sparse edge representation
structure CircuitEdge where
  source : Nat
  target : Nat
  weight : Float
```

### Incomplete Theorems (Need Lean Expert)

1. `property_transfer` - Core value proposition
2. `lipschitz_composition_bound` - Error bound justification
3. `circuit_robustness_example` - Robustness proof
4. 13 other theorems with `sorry`

See [docs/PROOF_ROADMAP.md](docs/PROOF_ROADMAP.md) for full list.

---

## Common Patterns

### Adding to MBPP Benchmark

```python
# benchmarks/verina/run_benchmark.py (to be implemented)

def run_single_problem(problem_id: str, model_name: str) -> BenchmarkResult:
    """
    1. Load MBPP problem and Lean spec
    2. Have model generate solution
    3. Extract circuit from solution
    4. Translate to Lean
    5. Attempt to prove matches spec
    """
    pass
```

### Fixing `_evaluate_circuit()`

```python
# extraction/circuit_extractor.py

def _evaluate_circuit(self, circuit_components, inputs):
    """
    REQUIRED: Build sparse model and evaluate.

    1. Create new model with only circuit edges
    2. Apply masks to original weights
    3. Forward pass through sparse model
    4. Return sparse model output (NOT original model)
    """
    # TODO: Implement this properly
    pass
```

### Validating Lipschitz Tightness

```python
# scripts/validate_tightness.py (to be created)

def validate_tightness(circuit, test_data, model) -> float:
    """
    Check if theoretical bounds are reasonable.

    Returns ratio: theoretical_bound / empirical_max_error
    Target: ratio < 100x
    """
    pass
```

---

## Code Review Checklist

Before committing, verify:

- [ ] Relates to Martian challenge focus (code LLMs, circuits, proofs)
- [ ] All tests pass (`python -m pytest` and `lake build`)
- [ ] New code has tests
- [ ] Type hints/annotations complete
- [ ] Docstrings present
- [ ] No `sorry` added without justification
- [ ] No debug code left in
- [ ] Updates ROADMAP.md if completing a task

---

## References

- [Martian Interpretability Challenge](https://withmartian.com/prize)
- [VERINA/MBPP-Lean Benchmark](https://github.com/sunblaze-ucb/verina)
- [Lean 4 Documentation](https://leanprover.github.io/lean4/doc/)
- [Mathlib4 Docs](https://leanprover-community.github.io/mathlib4_docs/)
