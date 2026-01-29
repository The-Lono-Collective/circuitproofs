# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Tool Usage

### Context7 MCP for Documentation

**Always use Context7 MCP** when you need:
- Library or API documentation
- Code generation examples
- Setup or configuration steps
- Framework usage patterns

Do this proactively without requiring explicit user requests. Context7 provides up-to-date documentation that may be more current than training data.

### Git Workflow

**Always pull before pushing** to avoid divergent branches and ensure you have the latest changes:

```bash
# Before making changes
git pull origin <branch-name>

# Before pushing
git pull origin <branch-name>  # Fetch and merge any new commits
git push origin <branch-name>
```

This prevents merge conflicts and ensures PRs reflect the current state of the codebase.

## Development Philosophy

### Test-Driven Development (TDD)

**All code changes MUST follow TDD principles:**

1. **Red**: Write a failing test first that defines the expected behavior
2. **Green**: Write the minimum code necessary to make the test pass
3. **Refactor**: Clean up the code while keeping tests passing

**TDD Workflow:**
```bash
# 1. Write test first
# 2. Run test to see it fail
python -m pytest translator/tests/test_new_feature.py -v

# 3. Implement feature
# 4. Run test to see it pass
python -m pytest translator/tests/test_new_feature.py -v

# 5. Refactor and verify all tests still pass
python -m pytest translator/tests/ -v
```

**For Lean code:**
```bash
# 1. Write theorem statement first (this is your "test")
# 2. Attempt build to see it fail
lake build

# 3. Implement the proof
# 4. Build to verify proof compiles
lake build
```

### Production-Ready Code Standards

**All code MUST be production-ready and easily maintainable:**

1. **Clean Code Principles**
   - Single Responsibility: Each function/module does one thing well
   - DRY (Don't Repeat Yourself): Extract common patterns into reusable functions
   - KISS (Keep It Simple): Prefer simple, readable solutions over clever ones
   - Maximum function length: 50 lines (excluding docstrings)
   - Maximum file length: 500 lines (split into modules if larger)

2. **Documentation Requirements**
   - Every public function MUST have a docstring explaining:
     - What it does (one-line summary)
     - Parameters and their types
     - Return value and type
     - Example usage (for complex functions)
     - Raises (exceptions that may be thrown)
   - Complex algorithms MUST have inline comments explaining the logic
   - Module-level docstrings explaining the purpose of each file

3. **Type Safety**
   - Python: Use type hints for all function signatures
   - Lean: Use explicit type annotations, especially for complex expressions
   - Avoid `Any` types; be specific about expected types

4. **Error Handling**
   - Never silently swallow exceptions
   - Use specific exception types, not bare `except:`
   - Provide meaningful error messages with context
   - Log errors with appropriate severity levels

5. **Naming Conventions**
   - Use descriptive, intention-revealing names
   - Variables: `snake_case` (Python), `camelCase` (Lean)
   - Functions: `snake_case` (Python), `camelCase` (Lean)
   - Classes/Structures: `PascalCase`
   - Constants: `SCREAMING_SNAKE_CASE`
   - Avoid abbreviations unless universally understood

6. **Code Organization**
   - Group related functionality into modules
   - Use consistent file structure across the project
   - Separate concerns: data, logic, I/O, presentation
   - Keep imports organized and minimal

7. **Performance Considerations**
   - Profile before optimizing
   - Document performance-critical sections
   - Use appropriate data structures for the task
   - Consider memory usage for large-scale operations

8. **Security**
   - Never commit secrets, credentials, or API keys
   - Validate and sanitize all external inputs
   - Use parameterized queries for any database operations
   - Follow principle of least privilege

### Code Review Checklist

Before committing, verify:
- [ ] All tests pass (`python -m pytest` and `lake build`)
- [ ] New code has corresponding tests
- [ ] Type hints/annotations are complete
- [ ] Docstrings are present and accurate
- [ ] No debug code or print statements left in
- [ ] Error handling is appropriate
- [ ] Code follows project style conventions
- [ ] No security vulnerabilities introduced

## Project Overview

**LeanVerifier** (formerly FormalVerifML) is a formal verification framework for machine learning models using Lean 4. The key innovation is **Certified Proof-Carrying Circuits** - a novel pipeline that bridges mechanistic interpretability and formal verification by extracting sparse computational subgraphs (circuits) from neural networks and formally verifying their properties.

## Core Architecture

### Three-Component Pipeline

1. **Python Extraction Layer** (`extraction/`, `translator/`)
   - Extracts circuits from PyTorch models using BlockCert-style pruning
   - Translates PyTorch/HuggingFace models to JSON intermediate format
   - Generates Lean 4 code from JSON specifications
   - Computes certified error bounds via Lipschitz composition

2. **Lean 4 Verification Core** (`lean/FormalVerifML/`)
   - **`base/`**: Core mathematical definitions and properties
     - `definitions.lean`: Base ML model types (NeuralNet, LinearModel, DecisionTree)
     - `circuit_models.lean`: Circuit definitions with sparse edge representations
     - `ml_properties.lean`: Robustness, fairness, interpretability properties
     - `advanced_models.lean`: Transformer and attention mechanisms
     - `vision_models.lean`: Vision Transformer (ViT) architectures
     - `large_scale_models.lean`: 100M+ parameter models with distributed support
     - `enterprise_features.lean`: Multi-user, audit logging, rate limiting
   - **`generated/`**: Auto-generated model definitions from Python translator
   - **`proofs/`**: Verification proof scripts for specific properties

3. **Web Interface** (`webapp/`)
   - Flask-based UI for uploading models and visualizing verification results

### Key Data Flow

```
PyTorch Model → JSON → Lean Definition → Formal Proof → Certificate
     ↓             ↓          ↓              ↓
  export_from  generate   lake build    verification
  _pytorch.py  _lean_model              results
```

For circuits specifically:
```
PyTorch Model → Circuit Extraction → Circuit JSON → Lean Circuit → Proofs
                (circuit_extractor.py)  (circuit_to_lean.py)
```

## Development Commands

### Building and Testing

```bash
# Build all Lean code and run verification
lake build

# Build and run the main executable
lake exe formal_verif_ml_exe

# Clean build artifacts
lake clean

# Update Lean dependencies (mathlib)
lake update
```

### Python Translation Workflow

```bash
# Export PyTorch model to JSON
python translator/export_from_pytorch.py \
    --model_path <path> \
    --output_path model.json \
    --model_type [nn|transformer|vit]

# Generate Lean code from JSON
python translator/generate_lean_model.py \
    --model_json model.json \
    --output_lean lean/FormalVerifML/generated/my_model.lean

# Circuit extraction and translation
python extraction/circuit_extractor.py  # or use example_extraction.py
python translator/circuit_to_lean.py \
    --circuit_json circuit.json \
    --output_dir lean/FormalVerifML/generated
```

### Testing

```bash
# Run comprehensive Python tests
python translator/run_comprehensive_tests.py

# Run all tests with pytest
python -m pytest translator/tests/ -v --cov=translator

# Test enterprise features
python translator/test_enterprise_features.py

# Test HuggingFace model support
python translator/test_huggingface_models.py

# Run end-to-end circuit pipeline example
python examples/end_to_end_pipeline.py
```

### Web Interface

```bash
# Start Flask development server
python webapp/app.py

# Production deployment
gunicorn -w 4 -b 0.0.0.0:5000 webapp.app:app
```

### Docker

```bash
# Build Docker image
docker build -t circuitproofs .

# Run with volume mount
docker run -p 5000:5000 -v $(pwd)/models:/app/models circuitproofs
```

## Critical Architecture Notes

### Sparse vs Dense Representations

The circuit verification approach relies on **sparse edge-based representations** rather than dense weight matrices. This is crucial for tractability:

- Dense matrix: O(n²) verification complexity
- Sparse edges: O(k) complexity where k = number of non-zero weights

When working with circuits in Lean, always use `List CircuitEdge` (defined in `circuit_models.lean`) rather than dense arrays.

### Error Bound Certification

The BlockCert-style error bound computation uses **Lipschitz composition**:
- Each circuit component has local error ε_i and Lipschitz constant L_i
- Global error bound: `‖F̂(x) - F(x)‖ ≤ Σ_i (ε_i ∏_{j>i} L_j)`
- This bound is computed empirically during extraction and verified formally in Lean

See `extraction/circuit_extractor.py:compute_error_bounds()` for implementation.

### Model Type Hierarchy

The codebase supports multiple model types with different complexity levels:

1. **Basic**: LinearModel, DecisionTree (in `definitions.lean`)
2. **Neural Networks**: NeuralNet with LayerType (in `definitions.lean`)
3. **Transformers**: MultiHeadAttention, TransformerBlock (in `advanced_models.lean`)
4. **Vision**: VisionTransformer, PatchEmbedding (in `vision_models.lean`)
5. **Circuits**: SparseCircuit, CircuitComponent (in `circuit_models.lean`)

When adding new model types, follow the pattern in `definitions.lean` with mathematical structures and evaluation functions.

### Generated Code Integration

The `lean/FormalVerifML/generated/` directory contains auto-generated Lean definitions. The main entry point (`formal_verif_ml.lean`) imports these files. When adding new models:

1. Generate Lean code via `generate_lean_model.py` or `circuit_to_lean.py`
2. Add import statement to `formal_verif_ml.lean`
3. Optionally add theorem to verify the model type-checks
4. Run `lake build` to verify compilation

### Lean 4 Version Pinning

The project uses Lean 4 v4.18.0-rc1 (specified in `lean-toolchain`). Do not change this without testing all verification proofs, as Lean 4 API changes can break proofs.

### Dependencies

- **Lean**: Depends on mathlib4 (specified in `lakefile.lean`)
- **Python**: See `translator/requirements.txt` for PyTorch, transformers, Flask, etc.
- **Key libraries**: torch, transformers, numpy, scikit-learn (for ML), flask, gunicorn (for web)

## Common Development Patterns

### Adding a New Verification Property

1. **Write tests first** (TDD):
   - Add test case in `translator/tests/` for Python functionality
   - Write theorem statement in Lean before implementing proof
2. Define the property in `lean/FormalVerifML/base/ml_properties.lean`
3. Add proof template in `lean/FormalVerifML/proofs/`
4. Import in `formal_verif_ml.lean`
5. Verify all tests pass

### Supporting a New Model Architecture

1. **Write tests first** (TDD):
   - Create test cases for model parsing and translation
   - Define expected Lean output format
2. Add architecture parsing to `translator/export_from_pytorch.py`
3. Update JSON schema if needed
4. Add Lean definitions to appropriate base file (or create new one)
5. Update `generate_lean_model.py` to handle new model type
6. Add example model in `translator/*.json` for testing
7. Run full test suite to verify

### Debugging Lean Verification

- Use `#check` and `#eval` commands in Lean for type checking and evaluation
- Run `lake build` to see compilation errors
- Check generated code in `lean/FormalVerifML/generated/` for issues
- Verify JSON is well-formed before attempting Lean code generation

### Working with Circuits

The circuit extraction pipeline requires calibration data:
- Use small representative dataset (100-1000 examples)
- Pruning threshold controls sparsity (0.01 = ~70-90% sparse)
- Always verify error bounds are acceptable for your use case
- See `examples/end_to_end_pipeline.py` for complete workflow

## Important File Relationships

- `lakefile.lean` defines the Lean package structure and dependencies
- `lean/FormalVerifML/formal_verif_ml.lean` is the main entry point that imports all modules
- `translator/generate_lean_model.py` generates code that must conform to types in `lean/FormalVerifML/base/`
- `webapp/app.py` uses the translator modules to provide web interface functionality

## Enterprise Features

The codebase includes production-grade features (implemented in `enterprise_features.lean` and tested in `test_enterprise_features.py`):

- Multi-user authentication and session management
- Role-based access control (RBAC)
- Audit logging with 90-day retention
- Rate limiting (100 requests/minute default)
- Distributed verification across multiple nodes
- Memory optimization for large-scale models (100M+ parameters)

These features are configurable via `EnterpriseConfig` structures in Lean.

## Testing Philosophy

- **Test-Driven Development**: Write tests before implementation
- **Python tests**: Focus on model loading, JSON generation, and end-to-end translation
- **Lean verification**: Proves mathematical properties using theorem prover
- **Integration tests**: Verify complete pipeline from PyTorch to formal proof
- **Coverage target**: 90%+ test coverage on Python code
- **Regression tests**: Every bug fix must include a test that would have caught it

## Documentation

- `README.md`: High-level overview and quick start
- `docs/CERTIFIED_CIRCUITS.md`: Detailed circuit verification methodology
- `docs/user_guide.md`: User-facing documentation
- `docs/developer_guide.md`: Architecture and extension guide
- `CONTRIBUTING.md`: Development setup and code standards
