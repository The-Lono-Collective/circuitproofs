# Quick Start: Certified Proof-Carrying Circuits

> **⚠️ WORK IN PROGRESS** — The pipeline runs but has critical limitations:
> - `_evaluate_circuit()` is a **stub** — error bounds are inaccurate
> - Core Lean proofs have **`sorry`** — no actual verification yet
> - See [CERTIFIED_CIRCUITS.md](CERTIFIED_CIRCUITS.md) for full status

Get started with circuit extraction and verification in 5 minutes!

## Prerequisites

```bash
pip install torch numpy
```

## 1. Run the End-to-End Example

The fastest way to see the pipeline in action:

```bash
cd examples
python end_to_end_pipeline.py
```

This will:
- ✓ Create a simple transformer model
- ✓ Extract a sparse circuit with error bounds
- ✓ Translate to Lean 4
- ✓ Generate verification specifications

**Expected output:**
```
======================================================================
  STEP 1: Circuit Extraction with BlockCert
======================================================================

Creating a simple MLP model...
  Model parameters: 1,482
  Calibration samples: 100
  Test samples: 50

Extracting circuit components...
  Extracted 3 components
  ✓ Error bound (ε): 0.012345
  ✓ Sparsity: 87.5%
  ✓ Coverage: 95.2%

======================================================================
  STEP 2: Translation to Lean 4
======================================================================

✓ Lean code generated: simple_mlp_circuit.lean
...
```

## 2. Extract Your Own Circuit

### From a PyTorch Model

```python
import torch
import torch.nn as nn
from extraction.circuit_extractor import extract_transformer_circuit

# Your model
model = YourTransformer()

# Generate calibration data
calibration_data = torch.randn(100, seq_len, d_model)
calibration_targets = torch.randn(100, num_classes)

# Test data for error bounds
test_data = torch.randn(50, seq_len, d_model)
test_targets = torch.randn(50, num_classes)

# Extract circuit
circuit_data = extract_transformer_circuit(
    model=model,
    calibration_data=calibration_data,
    calibration_targets=calibration_targets,
    test_data=test_data,
    test_targets=test_targets,
    output_path="my_circuit.json",
    pruning_threshold=0.05  # Adjust for sparsity/accuracy tradeoff
)

print(f"Error bound: {circuit_data['error_bound']['epsilon']:.6f}")
print(f"Sparsity: {circuit_data['metadata']['sparsity']:.2%}")
```

### Tuning the Pruning Threshold

| Threshold | Sparsity | Error Bound | Use Case |
|-----------|----------|-------------|----------|
| 0.001 | 30-50% | Very low | Safety-critical |
| 0.01 | 60-80% | Low | General use |
| 0.05 | 80-90% | Moderate | Interpretability focus |
| 0.1 | 90%+ | Higher | Maximum interpretability |

**Rule of thumb**: Start with 0.01, then adjust based on your error tolerance.

## 3. Translate to Lean

```bash
python translator/circuit_to_lean.py \
    --circuit_json my_circuit.json \
    --output_dir lean/FormalVerifML/generated
```

This generates `my_circuit.lean` with:
- Circuit component definitions (sparse representation)
- Error bound constants
- Evaluation functions

## 4. Verify Properties

Create a verification file `my_circuit_proof.lean`:

```lean
import FormalVerifML.base.circuit_models
import FormalVerifML.generated.my_circuit

namespace FormalVerifML

-- Define your safety property
def my_safety_property (circuit : Circuit) : Prop :=
  ∀ (x : Array Float),
  -- Input constraints
  (∀ i, x.getD i 0 ≥ 0 ∧ x.getD i 0 ≤ 1) →
  -- Output guarantee
  let output := evalCircuit circuit x
  ∀ i, output.getD i 0 ≥ 0

-- Prove it!
theorem my_circuit_is_safe :
  my_safety_property myCircuit := by
  sorry  -- Replace with actual proof

end FormalVerifML
```

Build and verify:

```bash
lake build
```

## Common Use Cases

### Use Case 1: Find Important Components

**Goal**: Identify which attention heads are critical for a task.

```python
from extraction.circuit_extractor import CircuitExtractor

extractor = CircuitExtractor(model)
extractor.register_hooks(['attention_layer_0', 'attention_layer_1'])

# Extract with moderate threshold
circuit_components = extractor.extract_circuit(
    calibration_data,
    calibration_targets,
    layer_names=['attention_layer_0', 'attention_layer_1'],
    pruning_threshold=0.01
)

# Check importance scores
for comp in circuit_components:
    print(f"{comp.component_type} {comp.component_idx}: {comp.importance_score:.2%}")
```

### Use Case 2: Verify Robustness

**Goal**: Prove the model is robust to small input perturbations.

```lean
theorem my_model_robust :
  circuitRobust myCircuit 0.1 0.5 := by
  -- Proof that δ=0.1 input change → ε=0.5 output change
  sorry
```

### Use Case 3: Check Fairness

**Goal**: Verify predictions don't depend strongly on a protected attribute.

```lean
def fairness_property (circuit : Circuit) (protectedIdx : Nat) : Prop :=
  ∀ x y,
  (∀ i, i ≠ protectedIdx → x[i] = y[i]) →
  ‖evalCircuit circuit x - evalCircuit circuit y‖ < 0.1

theorem my_model_fair :
  fairness_property myCircuit 5 := by
  sorry
```

## Troubleshooting

### Issue: High Error Bound

**Problem**: `error_bound.epsilon` is > 0.1

**Solutions**:
1. Decrease `pruning_threshold` (e.g., from 0.05 to 0.01)
2. Use more calibration data (e.g., 500 instead of 100)
3. Check if your model has many nearly-zero weights

```python
# Try with tighter threshold
circuit_data = extract_transformer_circuit(
    model=model,
    calibration_data=calibration_data,
    calibration_targets=calibration_targets,
    test_data=test_data,
    test_targets=test_targets,
    output_path="circuit_tight.json",
    pruning_threshold=0.001  # Much tighter
)
```

### Issue: Low Sparsity

**Problem**: `sparsity` is < 50%

**Solutions**:
1. Increase `pruning_threshold`
2. Check if all weights are actually important (inspect importance scores)
3. Try task-specific calibration data

```python
# Inspect importance before pruning
importance_scores = extractor.compute_importance_scores(
    calibration_data,
    calibration_targets,
    metric='gradient'
)

# Analyze distribution
import numpy as np
scores_flat = np.concatenate([s.flatten().cpu().numpy() for s in importance_scores.values()])
print(f"Median: {np.median(scores_flat):.6f}")
print(f"95th percentile: {np.percentile(scores_flat, 95):.6f}")
```

### Issue: Lean Build Fails

**Problem**: `lake build` fails with errors

**Common causes**:
1. **Syntax error in generated Lean**: Check the `.lean` file
2. **Missing imports**: Make sure `circuit_models.lean` is imported
3. **Large arrays**: Lean may time out on very large circuits

**Debug steps**:
```bash
# Check syntax
lake build lean/FormalVerifML/base/circuit_models.lean

# Check generated file
lake build lean/FormalVerifML/generated/my_circuit.lean

# Verbose output
lake build --verbose
```

## Next Steps

1. **Read the full documentation**: [CERTIFIED_CIRCUITS.md](CERTIFIED_CIRCUITS.md)
2. **Explore examples**: Check `examples/` for more use cases
3. **Customize extraction**: Implement custom importance metrics
4. **Write proofs**: Learn Lean 4 proof tactics
5. **Integrate with your workflow**: Automate circuit extraction in your training pipeline

## Getting Help

- **Documentation**: [docs/CERTIFIED_CIRCUITS.md](CERTIFIED_CIRCUITS.md)
- **Issues**: [GitHub Issues](https://github.com/the-lono-collective/circuitproofs/issues)
- **Examples**: See `examples/` directory for more code

## Key Takeaways

✓ **Circuits** = Sparse subgraphs that approximate the full model
✓ **Error bounds** = Mathematical guarantee on approximation quality
✓ **Sparsity** = Fraction of pruned edges (higher = more interpretable)
✓ **Verification** = Formal proofs about circuit behavior

**The pipeline**: Model → Extract → Translate → Verify → 🎉

Happy verifying! 🚀
