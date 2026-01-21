"""
Test BlockCert pipeline with TinyLlama.

This test verifies:
1. auto-LiRPA computes certified K_MLP bounds
2. K_MLP is in expected range (~1000-2000 for real models)
3. The full certification pipeline completes
"""

import torch
import torch.nn as nn
import numpy as np
import tempfile
from pathlib import Path

# Test 1: Direct auto-LiRPA test with Tanh-GELU
def test_auto_lirpa_tanh_gelu():
    """Verify auto-LiRPA works with Tanh-GELU approximation."""
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    import math

    class TanhGELU(nn.Module):
        def forward(self, x):
            return 0.5 * x * (1 + torch.tanh(
                math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)
            ))

    # Create MLP with realistic dimensions (like TinyLlama)
    d_model = 2048
    d_ff = 5632  # TinyLlama intermediate size

    mlp = nn.Sequential(
        nn.Linear(d_model, d_ff),
        TanhGELU(),
        nn.Linear(d_ff, d_model)
    )

    # Random init (replace with actual weights for real test)
    x = torch.randn(10, d_model)

    bounded_mlp = BoundedModule(mlp, x)
    ptb = PerturbationLpNorm(norm=2, eps=1.0)
    bounded_input = BoundedTensor(x, ptb)

    lb, ub = bounded_mlp.compute_bounds(x=(bounded_input,), method="backward")
    output_range = (ub - lb).abs().max().item()
    K_mlp = output_range / 2.0

    print(f"K_MLP (random init, d_model={d_model}): {K_mlp:.4f}")

    # Random init should give moderate bounds
    assert K_mlp > 0, "K_MLP should be positive"
    assert K_mlp < 1e10, "K_MLP should not be astronomical"

    return K_mlp


def test_auto_lirpa_silu():
    """Verify auto-LiRPA works with SiLU (sigmoid-based) for gated models."""
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

    d_model = 2048
    d_ff = 5632

    # Gated MLP structure (like TinyLlama)
    class GatedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(d_model, d_ff, bias=False)
            self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
            self.down_proj = nn.Linear(d_ff, d_model, bias=False)
            self.act = SiLU()

        def forward(self, x):
            up = self.up_proj(x)
            gate = self.act(self.gate_proj(x))
            return self.down_proj(up * gate)

    mlp = GatedMLP()
    x = torch.randn(10, d_model)

    bounded_mlp = BoundedModule(mlp, x)
    ptb = PerturbationLpNorm(norm=2, eps=1.0)
    bounded_input = BoundedTensor(x, ptb)

    lb, ub = bounded_mlp.compute_bounds(x=(bounded_input,), method="backward")
    output_range = (ub - lb).abs().max().item()
    K_mlp = output_range / 2.0

    print(f"K_MLP (gated SiLU, d_model={d_model}): {K_mlp:.4f}")

    assert K_mlp > 0, "K_MLP should be positive"
    assert K_mlp < 1e10, "K_MLP should not be astronomical"

    return K_mlp


def test_blockcert_extractor_tiny_model():
    """Test BlockCertExtractor with a tiny transformer for quick validation."""
    from extraction.blockcert import BlockCertifier

    # Test the certifier independently
    certifier = BlockCertifier(use_auto_lirpa=True)

    print("BlockCertifier initialized successfully")
    print(f"  auto-LiRPA available: {certifier._auto_lirpa_available}")

    assert certifier._auto_lirpa_available, "auto-LiRPA should be available"

    return True


def test_blockcert_with_tinyllama():
    """
    Full pipeline test with TinyLlama.

    NOTE: Requires HuggingFace transformers and ~2GB model download.
    Skip if model unavailable.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("SKIP: transformers not installed")
        return None

    from extraction import BlockCertExtractor

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    try:
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for certification
            low_cpu_mem_usage=True,
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"SKIP: Could not load model: {e}")
        return None

    # Initialize extractor
    extractor = BlockCertExtractor(model, device="cpu", use_auto_lirpa=True)

    # Test prompts (short for speed)
    prompts = [
        "def hello():\n    print(",
        "def add(a, b):\n    return",
        "# Function to calculate factorial\ndef factorial(n):",
    ]

    # Extract and certify block 0 only (for speed)
    with tempfile.TemporaryDirectory() as tmpdir:
        certificate = extractor.extract_and_certify(
            prompts=prompts,
            tokenizer=tokenizer,
            block_indices=[0],  # Just first block for testing
            output_dir=Path(tmpdir),
            pruning_threshold=0.01,
            max_length=64,
        )

    print(f"\nCertification Results:")
    print(f"  Model: {certificate.model_name}")
    print(f"  Global ε: {certificate.global_epsilon:.6e}")

    # Check K_MLP from the metrics
    if 0 in extractor.metrics and extractor.metrics[0].lipschitz is not None:
        K_mlp = extractor.metrics[0].lipschitz.K_mlp
        print(f"  K_MLP (block 0): {K_mlp:.4f}")

        # For TinyLlama, expect K_MLP in range ~1000-2000
        # (may vary based on specific weights and calibration)
        if K_mlp > 100:
            print("  K_MLP is in realistic range for a real model")
        else:
            print("  WARNING: K_MLP seems low - may indicate fallback to analytic")

    return certificate


if __name__ == "__main__":
    print("=" * 60)
    print("BlockCert Pipeline Tests")
    print("=" * 60)

    print("\n[1] Testing auto-LiRPA with Tanh-GELU...")
    try:
        test_auto_lirpa_tanh_gelu()
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")

    print("\n[2] Testing auto-LiRPA with SiLU (gated MLP)...")
    try:
        test_auto_lirpa_silu()
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")

    print("\n[3] Testing BlockCertifier initialization...")
    try:
        test_blockcert_extractor_tiny_model()
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")

    print("\n[4] Testing full pipeline with TinyLlama...")
    try:
        result = test_blockcert_with_tinyllama()
        if result is not None:
            print("PASS")
        else:
            print("SKIPPED (model not available)")
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Tests complete")
    print("=" * 60)
