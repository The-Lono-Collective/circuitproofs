"""
Lipschitz Bound Tightness Validation

Validates that theoretical error bounds from circuit extraction are reasonably
tight compared to empirical error measurements. This is a P0 gate for proceeding
to Phase 2 of the Martian Interpretability Challenge.

Target: theoretical_bound / empirical_max_error < 100x
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def validate_tightness(
    circuit: Dict,
    test_data: torch.Tensor,
    model: nn.Module,
    target_ratio: float = 100.0
) -> Tuple[float, bool, Dict]:
    """
    Check if theoretical bounds are reasonable.

    Args:
        circuit: Circuit dict with error_bound field
        test_data: Test input tensor for empirical evaluation
        model: Original model for comparison
        target_ratio: Maximum acceptable ratio (default 100x)

    Returns:
        Tuple of (ratio, is_valid, details_dict)
    """
    error_bound = circuit.get("error_bound", {})
    theoretical_bound = error_bound.get("epsilon", float("inf"))
    empirical_max_error = error_bound.get("max_error", 0.0)

    # Avoid division by zero
    if empirical_max_error == 0:
        empirical_max_error = 1e-10

    ratio = theoretical_bound / empirical_max_error
    is_valid = ratio < target_ratio

    details = {
        "theoretical_bound": theoretical_bound,
        "empirical_max_error": empirical_max_error,
        "ratio": ratio,
        "target_ratio": target_ratio,
        "is_valid": is_valid,
        "mae": error_bound.get("mae", None),
        "coverage": error_bound.get("coverage", None),
        "lipschitz_constants": error_bound.get("lipschitz_constants", [])
    }

    return ratio, is_valid, details


def validate_from_json(
    circuit_path: Path,
    target_ratio: float = 100.0
) -> Tuple[float, bool, Dict]:
    """
    Validate tightness from a circuit JSON file.

    Args:
        circuit_path: Path to circuit JSON file
        target_ratio: Maximum acceptable ratio

    Returns:
        Tuple of (ratio, is_valid, details_dict)
    """
    with open(circuit_path, "r") as f:
        circuit = json.load(f)

    return validate_tightness(circuit, None, None, target_ratio)


def compute_empirical_bound(
    model: nn.Module,
    circuit_model: nn.Module,
    test_data: torch.Tensor,
    device: str = "cpu"
) -> Dict:
    """
    Compute empirical error statistics between original and circuit model.

    Args:
        model: Original model
        circuit_model: Sparse circuit model
        test_data: Test input tensor
        device: Device to run on

    Returns:
        Dict with empirical statistics
    """
    model = model.to(device).eval()
    circuit_model = circuit_model.to(device).eval()
    test_data = test_data.to(device)

    with torch.no_grad():
        original_output = model(test_data)
        circuit_output = circuit_model(test_data)

        errors = torch.abs(original_output - circuit_output)

        return {
            "max_error": float(errors.max()),
            "mean_error": float(errors.mean()),
            "std_error": float(errors.std()),
            "median_error": float(errors.median()),
            "percentile_95": float(torch.quantile(errors.flatten(), 0.95)),
            "percentile_99": float(torch.quantile(errors.flatten(), 0.99))
        }


def analyze_lipschitz_product(lipschitz_constants: list) -> Dict:
    """
    Analyze how Lipschitz constants compound through layers.

    Large products indicate potentially vacuous bounds.
    """
    if not lipschitz_constants:
        return {"product": None, "geometric_mean": None, "max_constant": None}

    product = np.prod(lipschitz_constants)
    geometric_mean = np.exp(np.mean(np.log(lipschitz_constants)))
    max_constant = max(lipschitz_constants)

    return {
        "product": float(product),
        "geometric_mean": float(geometric_mean),
        "max_constant": float(max_constant),
        "num_layers": len(lipschitz_constants),
        "constants": lipschitz_constants
    }


def print_validation_report(details: Dict):
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("LIPSCHITZ BOUND TIGHTNESS VALIDATION REPORT")
    print("=" * 60)

    print(f"\nTheoretical bound (epsilon): {details['theoretical_bound']:.6f}")
    print(f"Empirical max error:         {details['empirical_max_error']:.6f}")
    print(f"Ratio:                       {details['ratio']:.2f}x")
    print(f"Target ratio:                < {details['target_ratio']:.0f}x")

    if details["is_valid"]:
        print(f"\n[PASS] Bounds are reasonably tight (ratio < {details['target_ratio']}x)")
    else:
        print(f"\n[FAIL] Bounds may be vacuous (ratio >= {details['target_ratio']}x)")

    if details.get("mae") is not None:
        print(f"\nMean absolute error: {details['mae']:.6f}")

    if details.get("coverage") is not None:
        print(f"Coverage (within bound): {details['coverage'] * 100:.1f}%")

    if details.get("lipschitz_constants"):
        lipschitz_analysis = analyze_lipschitz_product(details["lipschitz_constants"])
        print(f"\nLipschitz analysis:")
        print(f"  Product of constants: {lipschitz_analysis['product']:.2e}")
        print(f"  Geometric mean:       {lipschitz_analysis['geometric_mean']:.4f}")
        print(f"  Max single constant:  {lipschitz_analysis['max_constant']:.4f}")
        print(f"  Number of layers:     {lipschitz_analysis['num_layers']}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate Lipschitz bound tightness for extracted circuits"
    )
    parser.add_argument(
        "--circuit",
        type=str,
        required=True,
        help="Path to circuit JSON file"
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=100.0,
        help="Maximum acceptable ratio (default: 100)"
    )
    parser.add_argument(
        "--json-output",
        type=str,
        help="Optional path to write JSON results"
    )

    args = parser.parse_args()

    circuit_path = Path(args.circuit)
    if not circuit_path.exists():
        print(f"Error: Circuit file not found: {circuit_path}")
        sys.exit(1)

    ratio, is_valid, details = validate_from_json(circuit_path, args.target_ratio)

    print_validation_report(details)

    if args.json_output:
        output_path = Path(args.json_output)
        with open(output_path, "w") as f:
            json.dump(details, f, indent=2)
        print(f"\nResults written to: {output_path}")

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
