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

## 1. BlockCert Extraction API

The extraction pipeline is accessed through the BlockCert modules:

```python
from extraction.blockcert import BlockCertifier, BlockIR, BlockInterpreter, Certificate

# BlockCert provides:
# - BlockIR: Intermediate representation for transformer blocks
# - BlockInterpreter: Evaluates block computations
# - BlockCertifier: Certifies blocks with Lipschitz bounds
# - Certificate: Generates verification certificates
```

> **Note:** The legacy `end_to_end_pipeline.py` and `circuit_extractor.py` have been archived
> to the `archive/legacy-blockcert` branch. The SheafCert extraction pipeline (CD-T + DiscoGP)
> is planned but not yet implemented.

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

### Use Case 1: Verify Robustness

**Goal**: Prove the model is robust to small input perturbations.

```lean
theorem my_model_robust :
  circuitRobust myCircuit 0.1 0.5 := by
  -- Proof that δ=0.1 input change → ε=0.5 output change
  sorry
```

### Use Case 2: Check Fairness

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

Adjust BlockCertifier parameters for tighter certification bounds.

### Issue: Low Sparsity

**Problem**: `sparsity` is < 50%

**Solutions**:
1. Increase `pruning_threshold`
2. Check if all weights are actually important (inspect importance scores)
3. Try task-specific calibration data

Inspect BlockIR component weights to understand the importance distribution.

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
2. **Explore BlockCert modules**: See `extraction/blockcert/` for available tools
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
