# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Development Philosophy

### Test-Driven Development (TDD)

**All code changes MUST follow TDD:**

1. **Red**: Write a failing test that defines expected behavior
2. **Green**: Write minimum code to make the test pass
3. **Refactor**: Clean up while keeping tests passing

```bash
# Python workflow
python -m pytest translator/tests/test_new_feature.py -v  # 1. See it fail
# 2. Implement feature
python -m pytest translator/tests/ -v                      # 3. See it pass, refactor

# Lean workflow
lake build  # 1. See theorem fail (or use sorry placeholder)
# 2. Implement proof
lake build  # 3. See it compile, refactor
```

### Production Code Standards

| Requirement | Standard |
|-------------|----------|
| Function length | Max 50 lines (excluding docstrings) |
| File length | Max 500 lines (split into modules) |
| Type hints | Required for all function signatures |
| Docstrings | Required: summary, params, returns, raises |
| Error handling | No bare `except:`, meaningful messages |
| Naming | `snake_case` (Python funcs/vars), `camelCase` (Lean), `PascalCase` (classes) |

### Lean Verification Guidelines

**Never write vacuous proofs.** Theorems must verify actual properties, not just type-check.

```lean
-- BAD: Proves nothing (vacuous)
theorem foo : True :=
  let _ := someValue
  trivial

-- GOOD: Verifies actual property
theorem foo : circuitWellFormed myCircuit = true := by rfl

-- GOOD: Asserts condition holds
theorem bar : True :=
  let config := myConfig
  let _ : config.enabled = true := by native_decide
  trivial
```

**Proof methods by strength:**
1. `rfl` - Definitional equality (strongest, computed at compile-time)
2. `native_decide` - Decidable propositions (compile-time evaluation)
3. `decide` - Decision procedures
4. `sorry` - Placeholder only (**must have tracking issue**)

**File organization:**
- `formal_verif_ml.lean` - Library theorems and imports
- `Main.lean` - CLI executable (separate from library code)
- `proofs/*.lean` - Detailed mathematical proofs

## Project Overview

**LeanVerifier** formally verifies ML models using Lean 4. Key innovation: **Certified Proof-Carrying Circuits** - extracting sparse subgraphs from neural networks and proving their properties.

### Three-Component Pipeline

1. **Python Extraction Layer** (`extraction/`, `translator/`)
   - Extracts circuits from PyTorch using BlockCert-style pruning
   - Translates models to JSON intermediate format
   - Generates Lean 4 code from JSON

2. **Lean 4 Verification Core** (`lean/FormalVerifML/`)
   - `base/` - Core definitions (circuits, models, properties)
   - `generated/` - Auto-generated model definitions
   - `proofs/` - Verification proof scripts

3. **Web Interface** (`webapp/`) - Flask UI for model upload and visualization

### Key Data Flow

```
PyTorch Model → JSON → Lean Definition → Formal Proof → Certificate
     ↓             ↓          ↓              ↓
  export_from  generate   lake build    verification
  _pytorch.py  _lean_model              results
```

## Development Commands

```bash
# Lean build and run
lake build                       # Build all, run verification
lake exe formal_verif_ml_exe     # Run verification summary
lake clean                       # Clean build artifacts

# Python tests
python -m pytest translator/tests/ -v --cov=translator
python translator/run_comprehensive_tests.py

# Translation pipeline
python translator/export_from_pytorch.py --model_path <path> --output_path model.json
python translator/generate_lean_model.py --model_json model.json --output_lean <path>.lean

# Circuit extraction
python extraction/circuit_extractor.py
python translator/circuit_to_lean.py --circuit_json circuit.json --output_dir <dir>

# Web/Docker
python webapp/app.py                                    # Dev server
docker build -t circuitproofs . && docker run -p 5000:5000 circuitproofs
```

## Critical Architecture Notes

### Sparse vs Dense Representations

Circuits use **sparse edge-based representations** for tractability:
- Dense matrix: O(n²) verification complexity
- Sparse edges: O(k) complexity where k = non-zero weights

Always use `List CircuitEdge` (in `circuit_models.lean`), not dense arrays.

### Error Bound Certification

BlockCert-style error bounds use **Lipschitz composition**:
- Each component has local error ε_i and Lipschitz constant L_i
- Global bound: `‖F̂(x) - F(x)‖ ≤ Σᵢ (εᵢ ∏ⱼ₍ⱼ>ᵢ₎ Lⱼ)`

See `extraction/circuit_extractor.py:compute_error_bounds()`.

### Model Type Hierarchy

| Level | Types | Location |
|-------|-------|----------|
| Basic | LinearModel, DecisionTree | `definitions.lean` |
| Neural | NeuralNet, LayerType | `definitions.lean` |
| Transformer | MultiHeadAttention, TransformerBlock | `advanced_models.lean` |
| Vision | VisionTransformer, PatchEmbedding | `vision_models.lean` |
| Circuits | SparseCircuit, CircuitComponent | `circuit_models.lean` |

### Important Files

- `lakefile.lean` - Package structure, dependencies (mathlib4)
- `lean-toolchain` - Lean version pin (v4.18.0-rc1, don't change without testing)
- `formal_verif_ml.lean` - Main entry point, imports all modules
- `Main.lean` - Executable entry point (separate from library)

## Common Patterns

### Adding a Verification Property

1. Write theorem statement first (TDD)
2. Define property in `ml_properties.lean`
3. Add proof in `proofs/`
4. Import in `formal_verif_ml.lean`
5. Run `lake build`

### Adding a Model Architecture

1. Write test cases first (TDD)
2. Add parsing to `export_from_pytorch.py`
3. Add Lean definitions to appropriate base file
4. Update `generate_lean_model.py`
5. Run full test suite

### Debugging Lean

- `#check expr` - Type check expression
- `#eval expr` - Evaluate expression
- Check `generated/` for issues with auto-generated code
- Verify JSON is well-formed before Lean generation

## Before Committing

- [ ] `lake build` passes (no `sorry` without tracking issue)
- [ ] `python -m pytest` passes
- [ ] New code has corresponding tests
- [ ] Type hints and docstrings complete
- [ ] No debug code or print statements
