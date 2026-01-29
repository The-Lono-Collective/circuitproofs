# CircuitProofs Development Roadmap

**Target:** [Martian Interpretability Challenge](https://withmartian.com/prize) ($1M prize pool)

**Last Updated:** 2026-01-20

**Focus:** Formal verification of extracted circuits against ground-truth specifications

---

## Executive Summary

We are pivoting from a "General ML Verification Tool" to a **"Mechanistic Interpretability Microscope for Code LLMs"**. The winning angle: prove extracted circuits match formal specifications—not correlation, but mathematical certainty.

### Why This Wins

Martian criticizes current interpretability for being:
- **Not mechanistic** (correlation, not causation) → We provide Lean proofs
- **Not useful** (no real engineering tools) → We provide verification certificates
- **Incomplete** (narrow wins that don't generalize) → We test across multiple models
- **Doesn't scale** (enormous human effort) → We automate circuit extraction + verification

---

## Current State

### Overall Progress

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Critical Path Complete | 0% | 100% | Blocked by stubs and sorry |
| Lean Proofs Complete | 40% | 100% | 16 theorems have `sorry` |
| MBPP Benchmark | 10% | 100% | Scaffolding only |
| Test Pass Rate | 50% | 85% | Vision models failing (deprioritized) |

### Component Status

| Component | Status | Blocker |
|-----------|--------|---------|
| **A: Circuit Extraction** | ⚠️ 70% | `_evaluate_circuit()` is stub |
| **B: Circuit → Lean** | ✅ 85% | Minor issues |
| **C: Lean Proofs** | ❌ 40% | 16 `sorry` placeholders |
| **MBPP Benchmark** | ❌ 10% | Not implemented |
| **Distributed Infra** | ⚠️ 60% | Needs adaptation for extraction |

---

## Critical Blockers (Must Fix First)

### 1. SheafCert Extraction Pipeline (Not Yet Implemented)

**Location:** `extraction/blockcert/` (existing BlockCert modules); new SheafCert pipeline TBD

**Current State:** The legacy `circuit_extractor.py` has been archived to the `archive/legacy-blockcert` branch. The replacement SheafCert pipeline (CD-T + DiscoGP) has not been implemented.

**Required:** Implement the new extraction pipeline building on the existing BlockCert IR, interpreter, certifier, and certificate modules.

**Expert Needed:** MI/PyTorch engineer

### 2. Lipschitz Bound Tightness

**Risk:** Error bounds can explode through layer composition, making proofs vacuous.

**Validation Required:**
```python
# Before proceeding to Lean proofs
ratio = theoretical_bound / empirical_max_error
if ratio > 100:
    raise Error("Bounds too loose - fix extraction first")
```

**Gate:** Ratio must be < 100x before proceeding to Phase 2.

### 3. Core Lean Theorems

**`property_transfer`** - `lean/FormalVerifML/base/circuit_models.lean:217`
- Proves properties verified on circuits transfer to original model
- This is the core value proposition

**`lipschitz_composition_bound`** - `lean/FormalVerifML/base/circuit_models.lean:203`
- Justifies the error bound computation

**Expert Needed:** Lean expert (2 weeks total)

---

## Development Phases

### Phase 1: Critical Path (Weeks 1-2)

**Goal:** Unblock the core pipeline. Everything else depends on this.

| Task | Owner | Effort | Gate |
|------|-------|--------|------|
| Implement SheafCert extraction pipeline | MI engineer | TBD | New pipeline produces valid circuits |
| Run Lipschitz tightness validation | MI engineer | 1-2 days | **Ratio < 100x** |
| Complete `lipschitz_composition_bound` | Lean expert | 1 week | Compiles without sorry |
| Complete `property_transfer` theorem | Lean expert | 1 week | Compiles without sorry |

**Exit Criteria:**
- [ ] SheafCert extraction pipeline produces valid circuit output
- [ ] Tightness ratio < 100x on DeepSeek-Coder-1.3B test
- [ ] Core Lean theorems compile without `sorry`
- [ ] `lake build` passes

**Go/No-Go Decision (End of Week 1):**

Run tightness validation on DeepSeek-Coder-1.3B solving 3 MBPP problems.

| Result | Decision |
|--------|----------|
| Ratio < 10x | **GO** - proceed to Lean proofs |
| Ratio 10-100x | **CONDITIONAL** - investigate, may need tighter pruning |
| Ratio > 100x | **NO-GO** - fundamental issue, reassess approach |

### Phase 2: MBPP Benchmark Integration (Weeks 3-4)

**Goal:** Complete the ground-truth verification pipeline.

| Task | Owner | Effort | Depends On |
|------|-------|--------|------------|
| Implement `fetch_dataset.py` | Python eng | 1 day | — |
| Implement `run_benchmark.py` | Python eng | 2-3 days | fetch_dataset |
| Implement counterfactual variant testing | Python eng | 2 days | run_benchmark |
| Implement cross-model circuit comparison | Python eng | 2 days | run_benchmark |
| Test on DeepSeek-Coder-1.3B | ML eng | 3-4 days | All above |

**Exit Criteria:**
- [ ] Can fetch and parse MBPP-Lean dataset
- [ ] Can run extraction + verification on 10 MBPP problems
- [ ] ≥5/10 circuit proofs complete on DeepSeek-Coder-1.3B
- [ ] Counterfactual testing works (variable rename → same circuit)

### Phase 3: Scale & Generalize (Weeks 5-6)

**Goal:** Demonstrate the approach scales and generalizes.

| Task | Owner | Effort | Depends On |
|------|-------|--------|------------|
| Adapt distributed infra for extraction | Infra eng | 3-4 days | — |
| Run on StarCoder-7B | ML eng | 3-4 days | Phase 2 |
| Run on CodeLlama-34B | ML eng | 3-4 days | Distributed infra |
| Run on CodeLlama-70B | ML eng | 1 week | Distributed infra |
| Cross-model circuit comparison | ML eng | 2-3 days | Multiple model runs |

**Exit Criteria:**
- [ ] Successful extraction from 4 models (1.3B → 70B)
- [ ] Cross-model comparison shows similar circuits for same algorithms
- [ ] No OOM errors on 70B model

### Phase 4: Submission (Weeks 7-8)

**Goal:** Prepare compelling Martian challenge submission.

| Task | Owner | Effort |
|------|-------|--------|
| Analyze results, identify patterns | All | 3-4 days |
| Write technical report | Lead | 3-4 days |
| Create visualizations | ML eng | 2 days |
| Prepare code release | All | 2 days |
| Submit to Martian | Lead | 1 day |

**Deliverables:**
- Technical report showing verified circuits across multiple models
- Open-source codebase with reproducible results
- Lean proof artifacts

---

## Scope Decisions

### In Scope (Must Complete)

| Feature | Priority | Reason |
|---------|----------|--------|
| Circuit extraction (Component A) | P0 | Core pipeline |
| Lean translation (Component B) | P0 | Core pipeline |
| Core proofs (Component C) | P0 | Value proposition |
| MBPP-Lean benchmark | P0 | Ground truth |
| Distributed extraction | P1 | Scale to 70B |
| Cross-model comparison | P1 | Generalization |
| Counterfactual testing | P1 | Semantic verification |

### Out of Scope (Deprioritized)

| Feature | Previous Status | Decision | Reason |
|---------|-----------------|----------|--------|
| Vision models (ViT, CLIP) | 🟡 Partial | ❌ Deprioritize | Not code generation |
| Enterprise features | ✅ Complete | ❌ Deprioritize | Not needed for challenge |
| Web interface | ✅ Complete | ❌ Deprioritize | CLI sufficient |
| Generic HuggingFace | 🟡 Partial | ❌ Deprioritize | Focus on code LLMs |
| Multi-user auth | ✅ Complete | ❌ Deprioritize | Not needed |
| Audit logging | ✅ Complete | ❌ Deprioritize | Not needed |

### Keep But Rebrand

| Feature | Previous Purpose | New Purpose |
|---------|------------------|-------------|
| Distributed verification | General large models | Extraction from 70B code LLMs |
| Memory optimization | General efficiency | Handle activation caching |

---

## Risk Register

| Risk | Severity | Likelihood | Mitigation | Owner |
|------|----------|------------|------------|-------|
| **Lipschitz bound explosion** | Critical | Medium | Tightness gate Week 1 | MI eng |
| **Proofs harder than expected** | High | Medium | Start with simplest; get Lean help | Lean expert |
| **Circuits don't recover algorithms** | High | Medium | Start with simplest MBPP | ML eng |
| **70B extraction OOMs** | Medium | Medium | Distributed infra, checkpointing | Infra eng |
| **MBPP problems too simple** | Low | Low | Can extend to harder benchmarks | — |
| **Lean expert unavailable** | High | Unknown | Find backup; document progress | Lead |

---

## Team Requirements

| Role | Responsibilities | Availability Needed |
|------|------------------|---------------------|
| **Lean Expert** | Complete `sorry` theorems, review proofs | 2-3 weeks |
| **MI/PyTorch Engineer** | Fix `_evaluate_circuit()`, validate bounds | 2-3 weeks |
| **ML Engineer** | MBPP runner, model experiments | 4-5 weeks |
| **Distributed/Infra Engineer** | Adapt infra for 70B extraction | 1-2 weeks |
| **Project Lead** | Coordination, writing, submission | Throughout |

---

## Success Metrics

### Martian Challenge Alignment

| Martian Criterion | Our Approach | Measurable Target |
|-------------------|--------------|-------------------|
| **Mechanistic** | Lean proofs, not correlations | 100% proofs compile |
| **Useful** | Verification certificates | Usable by downstream tools |
| **Generalizable** | Cross-model testing | Same circuit structure across 3+ models |
| **Scalable** | 1.3B → 70B range | Successful runs on all 4 target models |
| **Ground truth benchmark** | MBPP-Lean specs | ≥50% proofs complete |
| **Code generation focus** | Only code LLMs | 100% target models are code-focused |

### Technical Milestones

| Milestone | Target Date | Criteria |
|-----------|-------------|----------|
| Tightness gate passed | Week 1 | Ratio < 100x |
| Core Lean proofs done | Week 2 | No `sorry` in P0 theorems |
| MBPP runner working | Week 4 | 10 problems extractable |
| Cross-model results | Week 6 | 3+ models compared |
| Submission ready | Week 8 | Report + code complete |

---

## Files to Modify/Create

### Must Modify

| File | Change | Priority |
|------|--------|----------|
| `extraction/blockcert/` | Implement SheafCert pipeline (CD-T + DiscoGP) | P0 |
| `lean/FormalVerifML/base/circuit_models.lean:203` | Complete `lipschitz_composition_bound` | P0 |
| `lean/FormalVerifML/base/circuit_models.lean:217` | Complete `property_transfer` | P0 |
| `lean/FormalVerifML/proofs/circuit_proofs.lean` | Complete all `sorry` | P0 |

### Must Create

| File | Purpose | Priority |
|------|---------|----------|
| `benchmarks/verina/fetch_dataset.py` | Download MBPP-Lean | P0 |
| `benchmarks/verina/run_benchmark.py` | Run extraction + verification | P0 |
| Counterfactual variant testing | Semantic verification | P1 |
| Cross-model circuit comparison | Generalization evidence | P1 |
| `scripts/validate_tightness.py` | Lipschitz bound validation | P0 |

### Can Ignore (Deprioritized)

| File/Directory | Reason |
|----------------|--------|
| `webapp/` | Web UI not needed |
| `archive/legacy-blockcert` branch | Archived: vision_models.lean, enterprise_features.lean, circuit_extractor.py, test files, etc. |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-20 | Pivoted to Martian challenge focus |
| 2026-01-20 | Added tightness gate from expert review |
| 2026-01-20 | Retained distributed infra (rebranded) |
| 2026-01-20 | Deprioritized vision, enterprise, web UI |
| 2026-01-16 | Initial roadmap created |
