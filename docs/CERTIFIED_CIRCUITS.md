# Certified Proof-Carrying Circuits

> **Status: Work In Progress** — Pipeline designed but core components not yet implemented.

## Overview

The pipeline integrates **CD-T** for rapid filtering, **DiscoGP** for sparse sheaf optimization, **Lean** for logical verification, and **BlockCert** for faithfulness bounding.

**Target:** [Martian Interpretability Challenge](https://withmartian.com/prize)

---

## Pipeline Architecture

```
Phase 1: Task Definition (D, s)
         ↓
Phase 2: Hybrid Extraction (CD-T → DiscoGP)
         ↓
Phase 3: Lean Verification (translation → convex relaxation → SMT)
         ↓
Phase 4: BlockCert Certification (local ε → Lipschitz composition → certificate)
```

---

## Phase 1: Adversarial Task Definition

We replace static datasets with a formal distribution to prevent overfitting and enable symbolic relaxation.

### The Task Tuple

We define the task $\tau = (\mathcal{D}, s)$.

- **$\mathcal{D}$ (The Prompt Distribution):** A generative grammar of adversarial prompts. For a type prediction task, we use Fill-In-The-Middle (FIM) templates where semantic shortcuts are ablated.
  - Let $T_{template}$ be a code skeleton. Let $V_{noise}$ be a set of randomized variable names (e.g., `__tmp0`, `var_x`).
  - The distribution is defined as $x \sim T_{template}(V_{noise})$.
- **$s$ (The Specification):** A binary scoring function acting on the logits.
  - $s(M(x)) = 1 \iff \text{Logit}_{\text{target}} - \max_{k \neq \text{target}} \text{Logit}_k > \delta$, where $\delta$ is a confidence margin.

### Datasets for Task Construction

| Dataset | Purpose |
|---------|---------|
| ManyTypes4Py | Adversarial FIM tasks (type prediction) |
| The Stack (TypeScript) | Adversarial FIM tasks (type prediction) |
| Verina (189 Lean 4 challenges) | Ground truth specifications |

---

## Phase 2: Hybrid Streaming Extraction

We solve the scalability bottleneck by cascading an analytical filter (CD-T) with a gradient-based pruner (DiscoGP).

### Step 2a: Coarse Filtering via Streaming CD-T

**Contextual Decomposition for Transformers (CD-T)** identifies relevant nodes (Heads/MLPs) without training.

**Decomposition:** For every activation $x$, decompose into relevant $\beta$ and irrelevant $\gamma$ such that $x = \beta + \gamma$.

- *Initialization:* $\beta$ is the embedding of the adversarial tokens, $\gamma$ is the mean embedding.
- *Linear Propagation:* $\beta_{out} = W\beta_{in} + b_{relevant}$
- *Non-Linear Propagation (ReLU/GeLU):*

$$\beta_{out} = W\beta_{in} + \frac{|W\beta_{in}|}{|W\beta_{in}| + |W\gamma_{in}|} \cdot b$$

**Relevance Score:** Relevance $R(s, T)$ of a source head $s$ to target logits $T$:

$$R(s, T) = \sum_{t \in T} \frac{\|\beta_t\|_{L1}}{\|\gamma_t\|_{L1}}$$

Prune all nodes where $R(s, T) < \tau_{cdt}$.

**Streaming:** Compute $\beta$ and $\gamma$ on the fly during a single forward pass batch, aggregate running mean of $R(s, T)$, discard activations immediately. No activation caching.

### Step 2b: Sheaf Refinement via Streaming DiscoGP

Optimize the weights and edges of the CD-T subgraph to create a **Sheaf**.

**Gumbel-Sigmoid Masks:** Assign learnable parameter $l_i$ to every weight/edge. Relax binary mask $m_i$:

$$s_i = \sigma \left( \frac{l_i - \log(-\log U_1) + \log(-\log U_2)}{\tau} \right)$$

$$m_i = \mathbb{I}_{s_i > 0.5} \text{ (forward)} + s_i \text{ (backward)}$$

**Optimization Objective:** Minimize joint loss $L_{GP}$:

$$L_{GP} = L_{fidelity} + \lambda_c L_{complete} + \lambda_s L_{sparse}$$

- **$L_{fidelity}$:** KL divergence between Sheaf and Full Model on $\mathcal{D}$.
- **$L_{sparse}$:** Sum of sigmoids to force sparsity:

$$L_{sparse} = \frac{1}{|m_\theta|} \sum \sigma(l_i) + \frac{1}{|m_E|} \sum \sigma(l_i)$$

**Result:** Sheaf $\hat{B}$ with <5% active parameters, dense $W$ replaced by sparse $W \odot m$.

### Calibration Baselines

Validate CD-T + DiscoGP against known circuits before applying to novel tasks:

| Baseline | Source | Purpose |
|----------|--------|---------|
| IOI (Indirect Object Identification) | TransformerLens | Known circuit structure |
| Greater-Than | TransformerLens | Known circuit structure |
| Tracr | DeepMind | Compiled circuits with ground truth |

---

## Phase 3: Lean Verification (Convex Relaxation & SMT)

Mathematically prove the logic of the Sheaf by checking the convex hull of the input space.

### Step 3a: Translation to Lean

- Use **Leanverifier** to transpile the sparse computational graph into Lean 4 definitions.
- **Compact Proof Strategy:** Decompose the Sheaf into QK circuit, OV circuit, and Direct Path. Approximate dense interactions via SVD low-rank approximations:

$$\text{QK}_{\text{approx}} = U \Sigma V^T + E_{error}$$

### Step 3b: Convex Relaxation

Define a **Convex Relaxation** $\mathcal{X}_{relaxed}$ of the input distribution $\mathcal{D}$.

- Instead of discrete tokens, define a continuous polytope in the embedding space containing all valid embeddings.
- **Theorem to Prove:** $\forall x \in \mathcal{X}_{relaxed}, \text{Sheaf}(x)_{\text{target}} > \text{Sheaf}(x)_{\text{other}}$

### Step 3c: SMT Solving (Pessimal Ablation)

The SMT solver verifies the theorem by pessimizing error terms.

- **Mean+Diff Trick:** Bound output logit difference $\Delta \ell$ by separating mean from variation:

$$\min \Delta \ell \geq \min (\text{Mean}) + \min (\text{Diff})$$

- **Max Row-Diff:** Bound matrix multiplication error:

$$\max_{i,j} ((AB)_{r,i} - (AB)_{r,j}) \leq \max_r \sum_k |A_{r,k}| \max_{i,j} (B_{k,i} - B_{k,j})$$

- **Outcome:** If solver returns `UNSAT` (no counter-example), the logic is verified.

---

## Phase 4: BlockCert Certification (Bounded Faithfulness)

Bound the "Dark Matter" (parts pruned in Phase 2) using **BlockCert**.

### Step 4a: Local Error Bounding ($\varepsilon$)

Calculate empirical error between Sheaf ($\hat{B}$) and Full Model ($B$) on $\mathcal{D}$:

$$\varepsilon_\ell = \max_{(p,t) \in \mathcal{D}} \| \hat{B}_\ell(x_{p,t}) - B_\ell(x_{p,t}) \|_2$$

Because DiscoGP minimizes $L_{fidelity}$, this $\varepsilon$ is minimized during extraction.

### Step 4b: Global Composition (Lipschitz)

Let $L_i$ be the Lipschitz constant of layer $i$. The global error bound:

$$\| \text{Sheaf}(x) - \text{Model}(x) \| \leq \sum_{i=0}^{L-1} \left( \varepsilon_i \prod_{j=i+1}^{L-1} L_j \right)$$

**Critical Risk:** Bounds can compound and become vacuous. Validation: `theoretical_bound / empirical_max_error` must be < 100x.

### Step 4c: The Certificate

Final artifact is a JSON file containing:

1. **Hashes:** SHA-256 of Sheaf weights and masks
2. **Logic Proof:** Lean 4 proof object from Phase 3 (outcome: `verified`)
3. **Faithfulness Bound:** Global error $\varepsilon_{global}$
4. **Guarantee:**

$$\text{Logit}_{\text{diff}}(\text{Sheaf}) > \varepsilon_{global} \implies \text{Logit}_{\text{diff}}(\text{Model}) > 0$$

If the circuit's margin of safety exceeds the maximum possible pruning error, the Model is guaranteed correct.

---

## Implementation Status

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

---

## Required Software

| Repository | Role |
|-----------|------|
| [princeton-nlp/Edge-Pruning](https://github.com/princeton-nlp/Edge-Pruning) | DiscoGP basis (gradient-based circuit pruning) |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | Hook-based CD-T implementation, IOI/Greater-Than baselines |
| [fraware/leanverifier](https://github.com/fraware/leanverifier) | PyTorch → Lean translation |
| [S3L-official/QEBVerif](https://github.com/S3L-official/QEBVerif) | MILP encoding reference for convex relaxation |
| [neuronpedia](https://www.neuronpedia.org/) | Feature visualization reference |

---

## References

1. **BlockCert**: Certified Approach to Mechanistic Interpretability
2. **CD-T**: Contextual Decomposition for Transformers
3. **DiscoGP**: Discovering General-Purpose Circuits (Edge-Pruning)
4. **Lean 4**: [The Lean Theorem Prover](https://leanprover.github.io/)
5. **Transformer Circuits**: [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/)
6. **VERINA**: [MBPP-Lean Benchmark](https://github.com/sunblaze-ucb/verina)
7. **Martian Challenge**: [Interpretability Prize](https://withmartian.com/prize)

---

**Last Updated**: January 2026
