# CircuitProofs: Formal Verification of Neural Network Circuits

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Lean 4](https://img.shields.io/badge/Lean-4-green.svg)](https://leanprover.github.io/)

> **Prove that extracted circuits match formal specifications** — not correlation, but mathematical certainty.

This project targets the [Martian Interpretability Challenge](https://withmartian.com/prize), addressing the core problem: current interpretability is correlational, not mechanistic. We provide **formal proofs** that extracted circuits implement specific algorithms.

---

## The Approach

Most interpretability work says: *"This circuit **seems** to compute X based on our analysis."*

We say: *"This circuit is **proven** to compute X. Here's the Lean proof."*

### Key Innovation: CD-T + DiscoGP + Convex Relaxation + BlockCert

We cascade four techniques into a single pipeline:

1. **CD-T** (Contextual Decomposition for Transformers) — analytical filter that identifies relevant heads/MLPs without training
2. **DiscoGP** (Discovering General-Purpose circuits) — gradient-based pruner that optimizes a sparse sheaf via Gumbel-Sigmoid masks
3. **Convex Relaxation + SMT** — formally verify the sheaf's logic over the convex hull of the input space
4. **BlockCert** — bound the faithfulness gap between sheaf and full model via Lipschitz composition

### Architecture

```
Phase 1: Adversarial Task Definition (D, s)
  FIM templates + binary scoring → task τ = (D, s)
         ↓
Phase 2: Hybrid Extraction (CD-T → DiscoGP)
  β/γ decomposition → relevance pruning → Gumbel-Sigmoid mask optimization → Sheaf
         ↓
Phase 3: Lean Verification
  Leanverifier translation → convex relaxation → SMT solving
         ↓
Phase 4: BlockCert Certification
  Local ε → Lipschitz composition → certificate JSON
  Guarantee: Logit_diff(Sheaf) > ε_global ⟹ Logit_diff(Model) > 0
```

---

## Project Status

| Phase | Component | Status | Location |
|-------|-----------|--------|----------|
| 1 | Task Definition | ❌ Not implemented | TBD |
| 2a | CD-T streaming | ❌ Not implemented | `extraction/` (planned) |
| 2b | DiscoGP optimization | ❌ Not implemented | `extraction/` (planned) |
| 3a | Lean translation | ⚠️ 85% | `translator/circuit_to_lean.py` |
| 3b | Convex relaxation | ❌ Not implemented | TBD |
| 3c | SMT solving | ❌ Not implemented | TBD |
| 4 | BlockCert certification | ⚠️ Partial | `extraction/blockcert/` |
| — | Lean proofs | ❌ 40% (16 sorry) | `lean/FormalVerifML/` |
| — | MBPP Benchmark | ❌ 10% | `benchmarks/verina/` |

**Critical Blockers:**
- CD-T + DiscoGP extraction not implemented
- Convex relaxation + SMT verification not implemented
- `property_transfer` theorem incomplete (core value proposition)
- `lipschitz_composition_bound` theorem incomplete
- MBPP benchmark runner not implemented

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

---

## Repository Structure

```
circuitproofs/
├── extraction/                 # Phase 2 & 4: Extraction + Certification
│   └── blockcert/             # BlockCert modules (IR, interpreter, certifier, certificate)
│   # Planned: cdt/ (CD-T streaming), discogp/ (sheaf optimization)
├── translator/                 # Phase 3a: Translation
│   ├── circuit_to_lean.py     # Circuit → Lean
│   └── generate_lean_model.py # Generic model → Lean
├── lean/FormalVerifML/        # Phase 3: Verification
│   ├── base/                  # Core definitions (complete)
│   ├── proofs/                # Theorems (16 sorry)
│   └── generated/             # Auto-generated models
├── benchmarks/verina/         # MBPP-Lean benchmark (scaffolding only)
├── examples/                  # Demo scripts
└── docs/                      # Documentation
```

---

## Development Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed phases.

---

## Known Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Lipschitz bound explosion** | Critical | Tightness gate: ratio < 100x |
| Proofs harder than expected | High | Start with simplest theorems |
| Circuits don't recover algorithms | High | Start with simple MBPP problems |
| 70B extraction OOMs | Medium | Distributed infrastructure |

---

## Contributing

See [CLAUDE.md](CLAUDE.md) for development standards, commands, and code review checklist.

---

## Attribution

- Extended from [FormalVerifML](https://github.com/fraware/formal_verif_ml)
- MBPP-Lean specifications from [VERINA](https://github.com/sunblaze-ucb/verina)
- Targeting [Martian Interpretability Challenge](https://withmartian.com/prize)

---

## License

MIT License - see [LICENSE](LICENSE)
