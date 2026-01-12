# LeanVerifier: Formal Verification of Machine Learning Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Lean 4](https://img.shields.io/badge/Lean-4-green.svg)](https://leanprover.github.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

> **Enterprise-grade formal verification framework for machine learning models with support for large-scale transformers, vision models, distributed verification, and novel Certified Proof-Carrying Circuits.**

> 🔬 **Note**: This is an extended fork of [FormalVerifML](https://github.com/fraware/formal_verif_ml) with added **Certified Proof-Carrying Circuits** capabilities that bridge mechanistic interpretability and formal verification.

<p align="center">
  <img src=".github/assets/FormalVerifML-RM.jpg" alt="FormalVerifML Logo" width="200"/>
</p>

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

LeanVerifier is a state-of-the-art framework for formally verifying machine learning models using Lean 4. It extends the original FormalVerifML with novel **Certified Proof-Carrying Circuits** that enable tractable verification of large models through sparse circuit extraction.

The framework provides comprehensive support for verifying properties such as robustness, fairness, interpretability, and safety across a wide range of model architectures.

### Mission

To provide **mathematically rigorous verification** of ML models for high-stakes applications in healthcare, finance, autonomous systems, and other critical domains where model reliability is paramount.

**Novel Contribution**: Bridge mechanistic interpretability and formal verification through certified sparse circuits with provable error bounds.

### What Makes This Fork Different

- **🔬 Certified Circuits**: Novel pipeline combining BlockCert-style extraction with formal verification
- **Mathematical Rigor**: Uses Lean 4 theorem prover for formal mathematical proofs
- **Tractable Verification**: 70-90% sparsity enables verification of previously intractable models
- **Production Ready**: Enterprise features with multi-user support, audit logging, and security
- **Scalable**: Supports models up to 100M+ parameters with distributed verification
- **Comprehensive**: Vision transformers, large-scale models, and advanced architectures
- **Automated**: SMT solver integration for automated proof generation

## Key Features

### 🔬 Certified Proof-Carrying Circuits (NEW!)

**Bridge interpretability and formal verification** with our novel pipeline that:
- **Extracts** sparse, interpretable circuits from neural networks
- **Certifies** error bounds using Lipschitz composition (BlockCert-style)
- **Verifies** safety properties on tractable circuit representations

See [Certified Circuits Documentation](docs/CERTIFIED_CIRCUITS.md) for details.

### Model Support

- **Neural Networks**: Feed-forward, convolutional, recurrent architectures
- **Transformers**: Full transformer support with multi-head attention
- **Vision Models**: ViT, Swin Transformers, CLIP-style multi-modal models
- **Large-Scale Models**: 100M+ parameter models with distributed processing
- **Decision Trees**: Interpretable tree-based models
- **Linear Models**: Logistic regression and linear classifiers
- **🆕 Circuits**: Sparse computational subgraphs with certified error bounds

### Verification Properties

- **Robustness**: Adversarial robustness and input perturbation resistance
- **Fairness**: Demographic parity, equalized odds, individual fairness
- **Interpretability**: Attention analysis, feature attribution verification
- **Safety**: Causal masking, sequence invariance, memory efficiency
- **Performance**: Memory optimization, distributed verification

### Enterprise Features

- **Multi-User Support**: Role-based access control and session management
- **Audit Logging**: Comprehensive activity tracking and compliance
- **Security**: Rate limiting, encryption, and input validation
- **Distributed Processing**: Multi-node verification with fault tolerance
- **Monitoring**: Real-time performance metrics and health checks

## Architecture

```
circuitproofs/
├── lean/                          # Lean 4 formal verification code
│   ├── FormalVerifML/
│   │   ├── base/                  # Core definitions and properties
│   │   │   ├── circuit_models.lean  # 🆕 Circuit definitions
│   │   │   └── ...
│   │   ├── generated/             # Auto-generated model definitions
│   │   └── proofs/                # Verification proof scripts
│   │       ├── circuit_proofs.lean  # 🆕 Circuit verification
│   │       └── ...
├── extraction/                    # 🆕 Circuit extraction from models
│   ├── circuit_extractor.py      # BlockCert-style extraction
│   └── example_extraction.py     # Example usage
├── translator/                    # Model translation and testing
│   ├── export_from_pytorch.py    # PyTorch model export
│   ├── generate_lean_model.py    # JSON to Lean code generation
│   ├── circuit_to_lean.py        # 🆕 Circuit to Lean translation
│   └── test_*.py                 # Comprehensive test suites
├── examples/                      # 🆕 End-to-end examples
│   └── end_to_end_pipeline.py    # Complete circuit pipeline
├── webapp/                       # Web interface and visualization
├── docs/                         # Documentation and guides
│   ├── CERTIFIED_CIRCUITS.md     # 🆕 Circuits documentation
│   └── ...
└── .github/                      # CI/CD and workflows
```

### Data Flow

1. **Model Export**: PyTorch/HuggingFace models → JSON format
2. **Code Generation**: JSON → Lean 4 definitions
3. **Verification**: Lean 4 → Formal proofs of properties
4. **Results**: Web interface visualization and reports

## Quick Start

### Prerequisites

- **Docker** (recommended) or **Python 3.9+** and **Lean 4**
- **8GB+ RAM** for large model verification
- **Modern web browser** for the interface

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/tektonics/circuitproofs.git
cd circuitproofs

# Build and run with Docker
docker build -t circuitproofs .
docker run -p 5000:5000 -v $(pwd)/models:/app/models circuitproofs

# Access the web interface
open http://localhost:5000
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/tektonics/circuitproofs.git
cd circuitproofs

# Install Python dependencies
pip install -r translator/requirements.txt

# Install Lean 4 (see https://leanprover.github.io/lean4/doc/setup.html)
# Then build the project
lake build

# Run the web interface
python webapp/app.py
```

## Usage

### Standard Workflow

#### 1. Export Your Model

```python
# Export a PyTorch model
python translator/export_from_pytorch.py \
    --model_path your_model.pth \
    --output_path model.json \
    --model_type transformer
```

#### 2. Generate Lean Code

```python
# Convert JSON to Lean definitions
python translator/generate_lean_model.py \
    --model_json model.json \
    --output_lean lean/FormalVerifML/generated/my_model.lean
```

#### 3. Verify Properties

```bash
# Build and verify with Lean
lake build
lake exe FormalVerifML
```

### 🔬 Certified Circuits Workflow

Extract interpretable circuits with certified error bounds:

```bash
# Run the complete pipeline
cd examples
python end_to_end_pipeline.py
```

Or step-by-step:

```python
# 1. Extract circuit from model
from extraction.circuit_extractor import extract_transformer_circuit

circuit_data = extract_transformer_circuit(
    model=your_model,
    calibration_data=calib_data,
    calibration_targets=calib_targets,
    test_data=test_data,
    test_targets=test_targets,
    output_path="circuit.json",
    pruning_threshold=0.01
)

# 2. Translate to Lean
python translator/circuit_to_lean.py \
    --circuit_json circuit.json \
    --output_dir lean/FormalVerifML/generated

# 3. Verify in Lean
lake build
```

**Key Benefits**:
- ✅ **Sparsity**: 70-90% reduction in parameters
- ✅ **Certified Bounds**: Mathematical guarantee on approximation error
- ✅ **Interpretability**: Human-understandable computational subgraphs
- ✅ **Efficient Verification**: Tractable proofs on sparse representations

See [full documentation](docs/CERTIFIED_CIRCUITS.md) for advanced usage.

### 4. Web Interface

Upload your model JSON files through the web interface at `http://localhost:5000` to:

- Visualize model architecture
- Generate Lean code automatically
- Run verification proofs
- View detailed logs and results

## Documentation

### User Guides

- **[User Guide](docs/user_guide.md)**: Getting started and basic usage
- **[Developer Guide](docs/developer_guide.md)**: Architecture and extension guide

### API Reference

- **[Lean API](lean/FormalVerifML/base/)**: Core definitions and properties
- **[Python API](translator/)**: Model translation and testing tools
- **[Web API](webapp/)**: Web interface and visualization

## Testing

### Run All Tests

```bash
# Comprehensive test suite
python translator/run_comprehensive_tests.py

# Enterprise feature tests
python translator/test_enterprise_features.py

# HuggingFace model tests
python translator/test_huggingface_models.py
```

### Test Categories

- ✅ **Model Loading**: PyTorch and HuggingFace model compatibility
- ✅ **Code Generation**: JSON to Lean translation accuracy
- ✅ **Verification**: Property verification correctness
- ✅ **Performance**: Memory usage and execution time
- ✅ **Enterprise**: Multi-user, security, and audit features

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/tektonics/circuitproofs.git
cd circuitproofs

# Install development dependencies
pip install -r translator/requirements.txt
pip install -r requirements-dev.txt

# Setup pre-commit hooks (if available)
pre-commit install

# Run tests
python translator/run_comprehensive_tests.py
```

### Code Standards

- **Python**: Follow PEP 8 with type hints
- **Lean**: Use Lean 4 style guide and mathlib conventions
- **Documentation**: Comprehensive docstrings and comments
- **Testing**: 90%+ test coverage required

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Lean Community**: For the excellent theorem prover
- **HuggingFace**: For transformer model support
- **PyTorch Team**: For the deep learning framework
- **Contributors**: All who have helped improve this project


## Attribution

This project extends [FormalVerifML](https://github.com/fraware/formal_verif_ml) with novel **Certified Proof-Carrying Circuits** capabilities.

**Original FormalVerifML**: Created with ❤️ by the FormalVerifML Team
**Certified Circuits Extension**: Developed by [tektonics](https://github.com/tektonics)

See [CHANGELOG_CIRCUITS.md](CHANGELOG_CIRCUITS.md) for details on the circuit verification additions.
