# MBPP-Lean Benchmark Integration

> **Status:** ❌ Not Implemented — Scaffolding only

This module will integrate the MBPP (Mostly Basic Python Problems) subset from the [VERINA benchmark](https://github.com/sunblaze-ucb/verina), which provides 49 MBPP problems manually translated to Lean 4 with formal specifications.

---

## Implementation Status

| File | Status | Description |
|------|--------|-------------|
| `README.md` | ✅ Complete | This documentation |
| `fetch_dataset.py` | ❌ **Not implemented** | Download VERINA dataset |
| `run_benchmark.py` | ❌ **Not implemented** | Run extraction + verification |
| `variant_generator.py` | ❌ **Not implemented** | Generate counterfactual variants |
| `circuit_comparator.py` | ❌ **Not implemented** | Compare circuits across models |
| `data/` | ❌ **Empty** | Downloaded tasks go here |

---

## Purpose

MBPP problems with Lean specifications allow us to:

1. **Validate Circuit Extraction**: Extract circuits from models solving MBPP problems, then verify the circuits match the formal Lean specifications
2. **Enable Counterfactual Testing**: Test that circuits capture semantics, not syntax, by comparing circuits from input variants (renamed variables, reordered arguments)
3. **Provide Ground Truth**: The pre-written Lean specifications serve as the "answer key" for what extracted circuits should compute

---

## Planned Architecture

```
Model solves MBPP problem (e.g., "calculate average")
         ↓
Extract circuit from model (Component A)
         ↓
Translate circuit to Lean (Component B)
         ↓
Prove circuit matches MBPP Lean specification (Component C)
         ↓
Certificate: Circuit correctly implements the algorithm
```

---

## Files to Implement

### 1. `fetch_dataset.py` (P0)

```python
"""
Download VERINA dataset from HuggingFace.
Extract MBPP-Lean subset (49 problems).
Save to data/mbpp/ directory.
"""

def fetch_verina_dataset() -> None:
    """Download and extract VERINA dataset."""
    # TODO: Implement
    # 1. Download from huggingface.co/datasets/sunblaze-ucb/verina
    # 2. Extract MBPP subset (verina-basic)
    # 3. Save to data/mbpp/
    pass

def load_mbpp_task(task_id: str) -> MBPPTask:
    """Load a single MBPP task with its Lean specification."""
    # TODO: Implement
    pass
```

### 2. `run_benchmark.py` (P0)

```python
"""
Run the full benchmark pipeline:
1. Load MBPP problem
2. Have model generate solution
3. Extract circuit
4. Translate to Lean
5. Attempt proof against spec
"""

def run_single_problem(
    problem_id: str,
    model_name: str
) -> BenchmarkResult:
    """Run benchmark on a single MBPP problem."""
    # TODO: Implement
    pass

def run_full_benchmark(
    model_name: str,
    problem_ids: List[str] = None
) -> List[BenchmarkResult]:
    """Run benchmark on all (or specified) MBPP problems."""
    # TODO: Implement
    pass
```

### 3. `variant_generator.py` (P1)

```python
"""
Generate semantic-preserving variants of MBPP problems.
Used for counterfactual testing.
"""

def generate_variants(problem: MBPPTask) -> List[MBPPTask]:
    """
    Generate variants:
    - Rename variables: `numbers` → `values`
    - Reorder parameters: `(a, b)` → `(b, a)`
    - Change formatting: whitespace, comments
    """
    # TODO: Implement
    pass
```

### 4. `circuit_comparator.py` (P1)

```python
"""
Compare circuits extracted from different models or variants.
Used to demonstrate generalization.
"""

def compare_circuits(
    circuit1: Circuit,
    circuit2: Circuit
) -> ComparisonResult:
    """
    Compare two circuits for structural similarity.
    Returns similarity score and differences.
    """
    # TODO: Implement
    pass

def compare_across_models(
    problem_id: str,
    model_names: List[str]
) -> CrossModelComparison:
    """Compare circuits from multiple models solving same problem."""
    # TODO: Implement
    pass
```

---

## Dataset Source

The MBPP-Lean specifications come from VERINA-BASIC, which contains:
- 49 MBPP problems translated from MBPP-DFY-50
- Each problem includes: natural language description, Lean implementation, formal specification, proof, and test cases
- Human-verified translations (not LLM-generated)

---

## Usage (After Implementation)

```bash
# Fetch the MBPP-Lean dataset
python benchmarks/verina/fetch_dataset.py

# Run benchmarks on MBPP subset
python benchmarks/verina/run_benchmark.py --model deepseek-coder-1.3b --subset mbpp

# Run with counterfactual testing
python benchmarks/verina/run_benchmark.py --model deepseek-coder-1.3b --counterfactual

# Compare across models
python benchmarks/verina/run_benchmark.py --models deepseek-coder-1.3b,starcoder-7b --compare
```

---

## Expected Directory Structure (After Implementation)

```
benchmarks/verina/
├── README.md              # This file
├── fetch_dataset.py       # Download VERINA dataset
├── run_benchmark.py       # Execute benchmark pipeline
├── variant_generator.py   # Generate input variants
├── circuit_comparator.py  # Compare circuits
└── data/                  # Downloaded tasks (gitignored)
    └── mbpp/              # MBPP-specific problems
        ├── task_001/
        │   ├── description.txt
        │   ├── solution.lean
        │   ├── specification.lean
        │   └── tests.lean
        ├── task_002/
        └── ...
```

---

## Integration with CircuitProofs

```python
# After implementation:
from benchmarks.verina import load_mbpp_task, verify_circuit_against_spec

# Load an MBPP task with its Lean specification
task = load_mbpp_task("task_001")

# Extract circuit from your model's solution
circuit = extract_circuit(model, task.input)

# Verify circuit matches the formal specification
result = verify_circuit_against_spec(circuit, task.lean_spec)
```

---

## References

- [VERINA Paper](https://arxiv.org/abs/2505.23135)
- [VERINA GitHub](https://github.com/sunblaze-ucb/verina)
- [VERINA Dataset](https://huggingface.co/datasets/sunblaze-ucb/verina)
- [Original MBPP](https://github.com/google-research/google-research/tree/master/mbpp)

---

## Priority

This is **P0** for the Martian challenge. Without the benchmark, we cannot demonstrate ground-truth verification.

See [ROADMAP.md](../../ROADMAP.md) Phase 2 for implementation timeline.
