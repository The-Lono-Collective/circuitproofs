/-
Main executable entry point for FormalVerifML.

This file is separate from the library code to:
1. Isolate CLI logic from verification logic
2. Prevent importing `main` when using FormalVerifML as a library
3. Provide accurate output about what has been verified

IMPORTANT: This file imports formal_verif_ml.lean which contains the
verification theorems. If ANY theorem fails to prove, this file will
not compile and the executable cannot be built. The verification
happens at COMPILE TIME, not runtime.
-/

import FormalVerifML.formal_verif_ml

open FormalVerifML

/--
Main entry point for the FormalVerifML executable.

This function explicitly witnesses each verified theorem. If any theorem
were invalid, this file would fail to compile due to the imports and
explicit references below.
-/
def main : IO Unit := do
  -- Explicitly witness each theorem to ensure they are in scope.
  -- If any of these theorems failed to prove, this file would not compile.
  let _ := project_builds_successfully
  let _ := sample_transformer_exists
  let _ := enterprise_features_integrated
  let _ := large_scale_models_integrated
  let _ := vision_models_integrated
  let _ := distributed_verification_integrated
  let _ := certified_circuits_integrated

  IO.println "╔══════════════════════════════════════════════════════════════════╗"
  IO.println "║           FormalVerifML - Verification Summary                   ║"
  IO.println "╚══════════════════════════════════════════════════════════════════╝"
  IO.println ""
  IO.println "The following properties were FORMALLY VERIFIED at compile time:"
  IO.println "(If any verification failed, this executable would not exist)"
  IO.println ""
  IO.println "┌─ Circuit Verification ─────────────────────────────────────────┐"
  IO.println "│ ✓ simpleLinearCircuit is well-formed (edges within bounds)     │"
  IO.println "│ ✓ Circuit evaluation produces correct output dimensions        │"
  IO.println "│   Theorem: certified_circuits_integrated                       │"
  IO.println "│   Proof method: rfl (definitional equality)                    │"
  IO.println "└────────────────────────────────────────────────────────────────┘"
  IO.println ""
  IO.println "┌─ Enterprise Features ─────────────────────────────────────────┐"
  IO.println "│ ✓ Authentication, encryption, audit logging enabled            │"
  IO.println "│ ✓ Audit retention ≥ 90 days                                    │"
  IO.println "│   Theorem: enterprise_features_integrated                      │"
  IO.println "│   Proof method: native_decide (compile-time evaluation)        │"
  IO.println "└────────────────────────────────────────────────────────────────┘"
  IO.println ""
  IO.println "┌─ Large-Scale Models ──────────────────────────────────────────┐"
  IO.println "│ ✓ Model parallelism and gradient checkpointing enabled         │"
  IO.println "│ ✓ Configured for ≥8 GPUs with 48 layers                        │"
  IO.println "│   Theorem: large_scale_models_integrated                       │"
  IO.println "│   Proof method: native_decide (compile-time evaluation)        │"
  IO.println "└────────────────────────────────────────────────────────────────┘"
  IO.println ""
  IO.println "┌─ Vision Transformers ─────────────────────────────────────────┐"
  IO.println "│ ✓ ViT-Base/16: 224×224 images, 16×16 patches, 12 heads         │"
  IO.println "│ ✓ Sequence length = 197 = (224/16)² + 1                        │"
  IO.println "│   Theorem: vision_models_integrated                            │"
  IO.println "│   Proof method: native_decide (compile-time evaluation)        │"
  IO.println "└────────────────────────────────────────────────────────────────┘"
  IO.println ""
  IO.println "┌─ Distributed Verification ────────────────────────────────────┐"
  IO.println "│ ✓ Fault tolerance and load balancing enabled                   │"
  IO.println "│ ✓ Parallel SMT with ≥8 nodes                                   │"
  IO.println "│   Theorem: distributed_verification_integrated                 │"
  IO.println "│   Proof method: native_decide (compile-time evaluation)        │"
  IO.println "└────────────────────────────────────────────────────────────────┘"
  IO.println ""
  IO.println "┌─ Sample Transformer ──────────────────────────────────────────┐"
  IO.println "│ ✓ Valid structure with numLayers = 2                           │"
  IO.println "│   Theorem: sample_transformer_exists                           │"
  IO.println "│   Proof method: rfl (definitional equality)                    │"
  IO.println "└────────────────────────────────────────────────────────────────┘"
  IO.println ""
  IO.println "Proof methods used:"
  IO.println "  • rfl           Definitional equality - strongest guarantee"
  IO.println "  • native_decide Compile-time evaluation of decidable propositions"
  IO.println ""
  IO.println "For universal (∀) proofs, see: LON-89"
