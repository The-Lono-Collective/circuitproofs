"""
Circuit Comparator for Counterfactual Testing

Compares extracted circuits for structural equivalence to validate
that circuit extraction captures semantics, not syntax.

This module supports the counterfactual testing methodology where
semantically equivalent code variants should produce structurally
equivalent circuits.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ComparisonResult:
    """Result of comparing two circuits."""
    is_equivalent: bool
    similarity_score: float
    component_match: bool
    type_match: bool
    edge_match: bool
    details: Dict


def compare_circuits(
    circuit1: Dict,
    circuit2: Dict,
    tolerance: float = 0.01
) -> ComparisonResult:
    """
    Compare two circuits for structural equivalence.

    Two circuits are structurally equivalent if they have:
    - Same number of components
    - Same component types in same order
    - Same edge structure (connections)
    - Weights within tolerance

    Args:
        circuit1: First circuit dict
        circuit2: Second circuit dict
        tolerance: Maximum allowed difference in weights

    Returns:
        ComparisonResult with equivalence status and details
    """
    comp1 = circuit1.get("components", [])
    comp2 = circuit2.get("components", [])

    # Check component count
    component_match = len(comp1) == len(comp2)

    # Check types match
    types1 = [c.get("component_type") for c in comp1]
    types2 = [c.get("component_type") for c in comp2]
    type_match = types1 == types2

    # Check edge structure
    edge_match = True
    edge_details = []

    if component_match:
        for i, (c1, c2) in enumerate(zip(comp1, comp2)):
            edges1 = c1.get("edges", c1.get("mask", []))
            edges2 = c2.get("edges", c2.get("mask", []))

            match, detail = _compare_edge_structure(edges1, edges2, tolerance)
            edge_details.append({"component_idx": i, "match": match, "detail": detail})

            if not match:
                edge_match = False

    is_equivalent = component_match and type_match and edge_match
    similarity = compute_circuit_similarity(circuit1, circuit2)

    return ComparisonResult(
        is_equivalent=is_equivalent,
        similarity_score=similarity,
        component_match=component_match,
        type_match=type_match,
        edge_match=edge_match,
        details={
            "component_count": (len(comp1), len(comp2)),
            "types": (types1, types2),
            "edge_details": edge_details,
            "tolerance": tolerance
        }
    )


def _compare_edge_structure(
    edges1,
    edges2,
    tolerance: float
) -> Tuple[bool, str]:
    """
    Compare edge structures of two components.

    Returns:
        Tuple of (match_bool, detail_string)
    """
    # Handle different edge representations
    if isinstance(edges1, list) and isinstance(edges2, list):
        if len(edges1) != len(edges2):
            return False, f"Edge count mismatch: {len(edges1)} vs {len(edges2)}"

        # If edges are dicts with source/target/weight
        if edges1 and isinstance(edges1[0], dict):
            sorted1 = sorted(
                edges1,
                key=lambda e: (e.get("sourceIdx", 0), e.get("targetIdx", 0))
            )
            sorted2 = sorted(
                edges2,
                key=lambda e: (e.get("sourceIdx", 0), e.get("targetIdx", 0))
            )

            for e1, e2 in zip(sorted1, sorted2):
                if e1.get("sourceIdx") != e2.get("sourceIdx"):
                    return False, f"Source mismatch: {e1.get('sourceIdx')} vs {e2.get('sourceIdx')}"
                if e1.get("targetIdx") != e2.get("targetIdx"):
                    return False, f"Target mismatch: {e1.get('targetIdx')} vs {e2.get('targetIdx')}"

                w1 = e1.get("weight", 0)
                w2 = e2.get("weight", 0)
                if abs(w1 - w2) > tolerance:
                    return False, f"Weight difference {abs(w1-w2):.4f} exceeds tolerance {tolerance}"

        # If edges are raw weight arrays
        elif edges1 and isinstance(edges1[0], (int, float)):
            for i, (w1, w2) in enumerate(zip(edges1, edges2)):
                if abs(w1 - w2) > tolerance:
                    return False, f"Weight[{i}] difference {abs(w1-w2):.4f} exceeds tolerance"

    return True, "Match"


def compute_circuit_similarity(circuit1: Dict, circuit2: Dict) -> float:
    """
    Compute a similarity score between two circuits.

    Returns:
        Float between 0 and 1, where 1 means identical structure
    """
    comp1 = circuit1.get("components", [])
    comp2 = circuit2.get("components", [])

    if not comp1 or not comp2:
        return 0.0

    # Component count similarity (40% weight)
    count_sim = min(len(comp1), len(comp2)) / max(len(comp1), len(comp2))

    # Type similarity (30% weight)
    types1 = set(c.get("component_type") for c in comp1)
    types2 = set(c.get("component_type") for c in comp2)
    common_types = len(types1 & types2)
    type_sim = common_types / max(len(types1), len(types2), 1)

    # Edge count similarity (30% weight)
    edges1 = sum(len(c.get("edges", c.get("mask", []))) for c in comp1)
    edges2 = sum(len(c.get("edges", c.get("mask", []))) for c in comp2)
    edge_sim = min(edges1, edges2) / max(edges1, edges2, 1)

    return 0.4 * count_sim + 0.3 * type_sim + 0.3 * edge_sim


def compare_error_bounds(
    circuit1: Dict,
    circuit2: Dict,
    tolerance: float = 0.1
) -> Tuple[bool, Dict]:
    """
    Compare error bounds of two circuits.

    Args:
        circuit1: First circuit dict
        circuit2: Second circuit dict
        tolerance: Relative tolerance for bound comparison

    Returns:
        Tuple of (bounds_similar, details)
    """
    eb1 = circuit1.get("error_bound", {})
    eb2 = circuit2.get("error_bound", {})

    eps1 = eb1.get("epsilon", 0)
    eps2 = eb2.get("epsilon", 0)

    max_eps = max(abs(eps1), abs(eps2), 1e-10)
    relative_diff = abs(eps1 - eps2) / max_eps

    bounds_similar = relative_diff < tolerance

    return bounds_similar, {
        "epsilon1": eps1,
        "epsilon2": eps2,
        "relative_difference": relative_diff,
        "tolerance": tolerance,
        "mae1": eb1.get("mae"),
        "mae2": eb2.get("mae"),
        "max_error1": eb1.get("max_error"),
        "max_error2": eb2.get("max_error")
    }


def batch_compare(
    reference_circuit: Dict,
    variant_circuits: List[Dict],
    tolerance: float = 0.01
) -> List[ComparisonResult]:
    """
    Compare multiple variant circuits against a reference.

    Args:
        reference_circuit: The reference circuit to compare against
        variant_circuits: List of variant circuits
        tolerance: Weight tolerance for equivalence

    Returns:
        List of ComparisonResults
    """
    results = []
    for variant in variant_circuits:
        result = compare_circuits(reference_circuit, variant, tolerance)
        results.append(result)
    return results


def summarize_batch_comparison(results: List[ComparisonResult]) -> Dict:
    """
    Summarize batch comparison results.

    Args:
        results: List of ComparisonResults from batch_compare

    Returns:
        Summary statistics dict
    """
    if not results:
        return {"count": 0}

    equivalent_count = sum(1 for r in results if r.is_equivalent)
    similarities = [r.similarity_score for r in results]

    return {
        "count": len(results),
        "equivalent_count": equivalent_count,
        "equivalent_ratio": equivalent_count / len(results),
        "mean_similarity": np.mean(similarities),
        "min_similarity": np.min(similarities),
        "max_similarity": np.max(similarities),
        "std_similarity": np.std(similarities)
    }
