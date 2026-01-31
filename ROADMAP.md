# CircuitProofs Development Roadmap

**Target:** [Martian Interpretability Challenge](https://withmartian.com/prize) ($1M prize pool)

**Last Updated:** 2026-01-31

**Focus:** Formal verification of extracted circuits against ground-truth specifications

---

## Current State

### Component Status

| Phase | Component | Status | Blocker |
|-------|-----------|--------|---------|
| 1 | Task Definition | ❌ 0% | Not implemented |
| 2a | CD-T streaming | ❌ 0% | Not implemented |
| 2b | DiscoGP optimization | ❌ 0% | Not implemented |
| 3a | Lean translation | ⚠️ 85% | Minor issues |
| 3b | Convex relaxation | ❌ 0% | Not implemented |
| 3c | SMT solving | ❌ 0% | Not implemented |
| 4 | BlockCert certification | ⚠️ 60% | Depends on extraction |
| — | Lean proofs | ❌ 40% | 16 `sorry` placeholders |
| — | MBPP Benchmark | ❌ 10% | Not implemented |

---

## Critical Blockers (Must Fix First)

### 1. CD-T Implementation

Implement streaming Contextual Decomposition for Transformers:
- $\beta$/$\gamma$ decomposition with streaming (no activation caching)
- Relevance scoring $R(s, T)$ with threshold $\tau_{cdt}$
- Build on TransformerLens hooks

### 2. DiscoGP Implementation

Implement gradient-based sheaf optimization:
- Gumbel-Sigmoid mask relaxation
- Joint loss $L_{GP} = L_{fidelity} + \lambda_c L_{complete} + \lambda_s L_{sparse}$
- Build on Edge-Pruning codebase

### 3. Convex Relaxation + SMT

Implement Phase 3 verification:
- Convex relaxation of input distribution into embedding polytope
- SMT solver with Mean+Diff trick and Max Row-Diff bound
- Reference: QEBVerif for MILP encoding

### 4. Lipschitz Bound Tightness

**Risk:** Error bounds can explode through layer composition, making proofs vacuous.

**Gate:** `theoretical_bound / empirical_max_error` must be < 100x before proceeding to Lean proofs.

### 5. Core Lean Theorems

**`property_transfer`** — `lean/FormalVerifML/base/circuit_models.lean:217`
- Proves properties verified on circuits transfer to original model

**`lipschitz_composition_bound`** — `lean/FormalVerifML/base/circuit_models.lean:203`
- Justifies the error bound computation

---

## Development Phases

### Phase 1: Extraction Pipeline

**Goal:** Implement CD-T + DiscoGP extraction, validate against known circuits.

| Task | Gate |
|------|------|
| Implement CD-T streaming decomposition | Produces relevance scores on IOI task |
| Implement DiscoGP mask optimization | Produces <5% sparse sheaf |
| Calibrate against IOI, Greater-Than, Tracr baselines | Recovers known circuit structure |
| Run Lipschitz tightness validation | **Ratio < 100x** on DeepSeek-Coder-1.3B |

**Go/No-Go Decision:**

| Result | Decision |
|--------|----------|
| Ratio < 10x | **GO** — proceed to Lean proofs |
| Ratio 10-100x | **CONDITIONAL** — investigate, may need tighter pruning |
| Ratio > 100x | **NO-GO** — fundamental issue, reassess approach |

### Phase 2: Verification + Benchmark

**Goal:** Complete verification pipeline and MBPP integration.

| Task | Depends On |
|------|------------|
| Implement convex relaxation of input space | Phase 1 |
| Implement SMT solving (Mean+Diff, Max Row-Diff) | Convex relaxation |
| Complete `lipschitz_composition_bound` | — |
| Complete `property_transfer` theorem | — |
| Implement `fetch_dataset.py` | — |
| Implement `run_benchmark.py` | fetch_dataset, extraction |
| Test on DeepSeek-Coder-1.3B with 10 MBPP problems | All above |

**Exit Criteria:**
- [ ] Convex relaxation + SMT produces verified/unverified outcomes
- [ ] Core Lean theorems compile without `sorry`
- [ ] Can run extraction + verification on 10 MBPP problems
- [ ] ≥5/10 circuit proofs complete on DeepSeek-Coder-1.3B

### Phase 3: Scale & Generalize

**Goal:** Demonstrate the approach scales and generalizes.

| Task | Depends On |
|------|------------|
| Run on StarCoder-7B | Phase 2 |
| Run on CodeLlama-34B | Distributed infra |
| Run on CodeLlama-70B | Distributed infra |
| Cross-model circuit comparison | Multiple model runs |

**Exit Criteria:**
- [ ] Successful extraction from 4 models (1.3B → 70B)
- [ ] Cross-model comparison shows similar circuits for same algorithms

### Phase 4: Submission

**Goal:** Prepare Martian challenge submission.

**Deliverables:**
- Technical report showing verified circuits across multiple models
- Open-source codebase with reproducible results
- Lean proof artifacts

---

## Files to Modify/Create

### Must Modify

| File | Change | Priority |
|------|--------|----------|
| `lean/FormalVerifML/base/circuit_models.lean:203` | Complete `lipschitz_composition_bound` | P0 |
| `lean/FormalVerifML/base/circuit_models.lean:217` | Complete `property_transfer` | P0 |
| `lean/FormalVerifML/proofs/circuit_proofs.lean` | Complete all `sorry` | P0 |

### Must Create

| File | Purpose | Priority |
|------|---------|----------|
| `extraction/cdt/` | CD-T streaming decomposition | P0 |
| `extraction/discogp/` | DiscoGP sheaf optimization | P0 |
| `extraction/verification/convex_relaxation.py` | Convex relaxation | P0 |
| `extraction/verification/smt_solver.py` | SMT solving | P0 |
| `benchmarks/verina/fetch_dataset.py` | Download MBPP-Lean | P0 |
| `benchmarks/verina/run_benchmark.py` | Run extraction + verification | P0 |
| `scripts/validate_tightness.py` | Lipschitz bound validation | P0 |

---

## Key Repositories

| Repository | Role |
|-----------|------|
| [princeton-nlp/Edge-Pruning](https://github.com/princeton-nlp/Edge-Pruning) | DiscoGP basis |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | CD-T hooks, calibration baselines |
| [fraware/leanverifier](https://github.com/fraware/leanverifier) | PyTorch → Lean translation |
| [S3L-official/QEBVerif](https://github.com/S3L-official/QEBVerif) | MILP encoding reference |
| [neuronpedia](https://www.neuronpedia.org/) | Feature visualization reference |

---

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Lipschitz bound explosion** | Critical | Tightness gate: ratio < 100x |
| **Proofs harder than expected** | High | Start with simplest theorems |
| **Circuits don't recover algorithms** | High | Calibrate against IOI/Greater-Than/Tracr first |
| **70B extraction OOMs** | Medium | Distributed infra, checkpointing |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-31 | Updated to 4-phase pipeline (CD-T + DiscoGP + convex relaxation + BlockCert) |
| 2026-01-20 | Pivoted to Martian challenge focus |
| 2026-01-20 | Added tightness gate from expert review |
| 2026-01-16 | Initial roadmap created |
