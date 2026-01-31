# Shared Review Context

This file contains project-specific rules that apply to all review tiers. Reference this in your review rather than duplicating these rules.

## Project: LeanVerifier

**Purpose**: Formal verification framework for ML models using Lean 4, with certified proof-carrying circuits.

## Lean Code Standards

**Must check for**:
- No `sorry` without tracking issue reference
- No vacuous proofs (patterns that prove `True` without meaningful assertions)
- Proper use of `rfl`, `native_decide`, `decide` tactics
- Theorems verify actual properties, not just type-check
- New theorems integrate with `formal_verif_ml.lean`

**Circuit proofs specifically**:
- Must use sparse edge representations (not dense matrices)
- Error bounds use Lipschitz composition
- Follow patterns in `circuit_models.lean`

## Python Code Standards

**Required**:
- Type hints on all function signatures
- Docstrings with: summary, params, returns, raises
- No bare `except:` blocks
- Max 50 lines per function
- Max 500 lines per file

**TDD workflow**:
- Tests must exist for new code
- Run `python -m pytest translator/tests/ -v` to verify

## Build Commands

```bash
# Lean
lake build        # Build and verify proofs

# Python
python -m pytest translator/tests/ -v
```

## File Risk Indicators

**High risk** (always review carefully):
- `lean/FormalVerifML/base/*.lean` - Core verification logic
- `translator/*.py` - Translation layer
- Any file with `security`, `auth`, `crypto` in path

**Medium risk**:
- `lean/FormalVerifML/generated/*.lean` - Generated code
- `tests/*.py` - Test files

**Low risk**:
- `*.md` documentation files
- Config files (unless secrets-related)
