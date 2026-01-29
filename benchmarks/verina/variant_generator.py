"""
Variant Generator for Counterfactual Testing

Generates semantically equivalent variants of MBPP tasks to test
whether circuit extraction captures true computational semantics
rather than surface-level patterns.
"""

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .fetch_dataset import MBPPTask


@dataclass
class TaskVariant:
    """A variant of an MBPP task with transformed code."""

    original_task_id: str
    variant_id: str
    variant_type: str  # "rename_vars", "reorder_params", "whitespace", etc.
    description: str
    lean_code: str
    lean_spec: str
    lean_proof: str
    test_cases: List[Dict]
    source: str
    transformations_applied: List[str]


def generate_variants(task: MBPPTask, variant_types: List[str] = None) -> List[TaskVariant]:
    """
    Generate semantically equivalent variants of a task.

    Args:
        task: Original MBPP task
        variant_types: List of variant types to generate. If None, generates all.
            Options: "rename_vars", "reorder_params", "add_comments",
                     "change_whitespace", "rename_function"

    Returns:
        List of TaskVariant objects
    """
    if variant_types is None:
        variant_types = [
            "rename_vars",
            "reorder_params",
            "add_comments",
            "change_whitespace"
        ]

    variants = []

    for vtype in variant_types:
        if vtype == "rename_vars":
            variants.extend(_generate_rename_variants(task))
        elif vtype == "reorder_params":
            variants.extend(_generate_reorder_variants(task))
        elif vtype == "add_comments":
            variants.extend(_generate_comment_variants(task))
        elif vtype == "change_whitespace":
            variants.extend(_generate_whitespace_variants(task))
        elif vtype == "rename_function":
            variants.extend(_generate_function_rename_variants(task))

    return variants


def _generate_rename_variants(task: MBPPTask) -> List[TaskVariant]:
    """Generate variants with renamed variables."""
    variants = []

    # Common variable rename mappings
    rename_maps = [
        {"numbers": "values", "number": "value", "nums": "vals", "num": "val"},
        {"list": "arr", "lst": "array", "items": "elements"},
        {"result": "output", "res": "out", "ret": "answer"},
        {"i": "idx", "j": "jdx", "k": "kdx"},
        {"x": "a", "y": "b", "z": "c"},
    ]

    for idx, rename_map in enumerate(rename_maps):
        new_code = task.lean_code
        new_spec = task.lean_spec
        new_proof = task.lean_proof
        transformations = []

        for old_name, new_name in rename_map.items():
            # Use word boundary matching to avoid partial replacements
            pattern = rf'\b{re.escape(old_name)}\b'

            if re.search(pattern, new_code):
                new_code = re.sub(pattern, new_name, new_code)
                new_spec = re.sub(pattern, new_name, new_spec)
                new_proof = re.sub(pattern, new_name, new_proof)
                transformations.append(f"{old_name}->{new_name}")

        if transformations:
            variant = TaskVariant(
                original_task_id=task.task_id,
                variant_id=f"{task.task_id}_rename_{idx}",
                variant_type="rename_vars",
                description=task.description,
                lean_code=new_code,
                lean_spec=new_spec,
                lean_proof=new_proof,
                test_cases=task.test_cases,
                source=task.source,
                transformations_applied=transformations
            )
            variants.append(variant)

    return variants


def _generate_reorder_variants(task: MBPPTask) -> List[TaskVariant]:
    """Generate variants with reordered function parameters."""
    variants = []

    # Find function definitions with multiple parameters
    # Pattern for Lean function definitions: def funcName (param1 : Type1) (param2 : Type2)
    func_pattern = r'def\s+(\w+)\s*(\([^)]+\)\s*)+:'

    matches = list(re.finditer(func_pattern, task.lean_code))

    for match in matches:
        # Extract parameters
        params_str = match.group(0)
        params = re.findall(r'\((\w+)\s*:\s*([^)]+)\)', params_str)

        if len(params) >= 2:
            # Create variant with first two params swapped
            swapped_params = list(params)
            swapped_params[0], swapped_params[1] = swapped_params[1], swapped_params[0]

            # Rebuild parameter string
            original_params_str = " ".join(f"({p[0]} : {p[1]})" for p in params)
            swapped_params_str = " ".join(f"({p[0]} : {p[1]})" for p in swapped_params)

            new_code = task.lean_code.replace(original_params_str, swapped_params_str)

            if new_code != task.lean_code:
                variant = TaskVariant(
                    original_task_id=task.task_id,
                    variant_id=f"{task.task_id}_reorder_0",
                    variant_type="reorder_params",
                    description=task.description,
                    lean_code=new_code,
                    lean_spec=task.lean_spec,  # Spec stays same - tests semantic equivalence
                    lean_proof=task.lean_proof,
                    test_cases=task.test_cases,
                    source=task.source,
                    transformations_applied=[f"swapped params {params[0][0]} and {params[1][0]}"]
                )
                variants.append(variant)

    return variants


def _generate_comment_variants(task: MBPPTask) -> List[TaskVariant]:
    """Generate variants with added comments."""
    comments_to_add = [
        "-- This is an auto-generated comment\n",
        "/- Multi-line\n   comment -/\n",
        "-- TODO: optimize this\n",
    ]

    variants = []

    for idx, comment in enumerate(comments_to_add):
        # Add comment at the beginning
        new_code = comment + task.lean_code

        variant = TaskVariant(
            original_task_id=task.task_id,
            variant_id=f"{task.task_id}_comment_{idx}",
            variant_type="add_comments",
            description=task.description,
            lean_code=new_code,
            lean_spec=task.lean_spec,
            lean_proof=task.lean_proof,
            test_cases=task.test_cases,
            source=task.source,
            transformations_applied=[f"added comment type {idx}"]
        )
        variants.append(variant)

    return variants


def _generate_whitespace_variants(task: MBPPTask) -> List[TaskVariant]:
    """Generate variants with different whitespace."""
    variants = []

    # Variant 1: Extra blank lines
    new_code_1 = re.sub(r'\n', r'\n\n', task.lean_code)
    variants.append(TaskVariant(
        original_task_id=task.task_id,
        variant_id=f"{task.task_id}_whitespace_0",
        variant_type="change_whitespace",
        description=task.description,
        lean_code=new_code_1,
        lean_spec=task.lean_spec,
        lean_proof=task.lean_proof,
        test_cases=task.test_cases,
        source=task.source,
        transformations_applied=["doubled newlines"]
    ))

    # Variant 2: Extra spaces around operators
    new_code_2 = re.sub(r'(\+|-|\*|/|:=|→|=>)', r' \1 ', task.lean_code)
    new_code_2 = re.sub(r'\s+', ' ', new_code_2)  # Normalize multiple spaces
    variants.append(TaskVariant(
        original_task_id=task.task_id,
        variant_id=f"{task.task_id}_whitespace_1",
        variant_type="change_whitespace",
        description=task.description,
        lean_code=new_code_2,
        lean_spec=task.lean_spec,
        lean_proof=task.lean_proof,
        test_cases=task.test_cases,
        source=task.source,
        transformations_applied=["extra spaces around operators"]
    ))

    return variants


def _generate_function_rename_variants(task: MBPPTask) -> List[TaskVariant]:
    """Generate variants with renamed functions."""
    variants = []

    # Find function names
    func_names = re.findall(r'def\s+(\w+)', task.lean_code)

    rename_suffixes = ["_v2", "_impl", "_new", "Helper"]

    for func_name in func_names:
        for suffix in rename_suffixes:
            new_name = func_name + suffix
            new_code = re.sub(rf'\b{func_name}\b', new_name, task.lean_code)
            new_spec = re.sub(rf'\b{func_name}\b', new_name, task.lean_spec)
            new_proof = re.sub(rf'\b{func_name}\b', new_name, task.lean_proof)

            variant = TaskVariant(
                original_task_id=task.task_id,
                variant_id=f"{task.task_id}_funcname_{suffix.strip('_')}",
                variant_type="rename_function",
                description=task.description,
                lean_code=new_code,
                lean_spec=new_spec,
                lean_proof=new_proof,
                test_cases=task.test_cases,
                source=task.source,
                transformations_applied=[f"renamed {func_name} to {new_name}"]
            )
            variants.append(variant)

    return variants


def compare_circuits(circuit1: Dict, circuit2: Dict, tolerance: float = 0.01) -> bool:
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
        True if circuits are structurally equivalent
    """
    comp1 = circuit1.get("components", [])
    comp2 = circuit2.get("components", [])

    # Check component count
    if len(comp1) != len(comp2):
        return False

    for c1, c2 in zip(comp1, comp2):
        # Check component type
        if c1.get("component_type") != c2.get("component_type"):
            return False

        # Check edge structure
        edges1 = c1.get("edges", c1.get("mask", []))
        edges2 = c2.get("edges", c2.get("mask", []))

        if not _compare_edge_structure(edges1, edges2, tolerance):
            return False

    return True


def _compare_edge_structure(edges1, edges2, tolerance: float) -> bool:
    """Compare edge structures of two components."""
    # Handle different edge representations
    if isinstance(edges1, list) and isinstance(edges2, list):
        if len(edges1) != len(edges2):
            return False

        # If edges are dicts with source/target/weight
        if edges1 and isinstance(edges1[0], dict):
            # Sort by source, target for comparison
            sorted1 = sorted(edges1, key=lambda e: (e.get("sourceIdx", 0), e.get("targetIdx", 0)))
            sorted2 = sorted(edges2, key=lambda e: (e.get("sourceIdx", 0), e.get("targetIdx", 0)))

            for e1, e2 in zip(sorted1, sorted2):
                if e1.get("sourceIdx") != e2.get("sourceIdx"):
                    return False
                if e1.get("targetIdx") != e2.get("targetIdx"):
                    return False
                if abs(e1.get("weight", 0) - e2.get("weight", 0)) > tolerance:
                    return False

        # If edges are raw weight arrays
        elif edges1 and isinstance(edges1[0], (int, float)):
            for w1, w2 in zip(edges1, edges2):
                if abs(w1 - w2) > tolerance:
                    return False

    return True


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

    # Component count similarity
    count_sim = min(len(comp1), len(comp2)) / max(len(comp1), len(comp2))

    # Type similarity
    types1 = [c.get("component_type") for c in comp1]
    types2 = [c.get("component_type") for c in comp2]
    common_types = len(set(types1) & set(types2))
    type_sim = common_types / max(len(set(types1)), len(set(types2)), 1)

    # Edge count similarity
    edges1 = sum(len(c.get("edges", [])) for c in comp1)
    edges2 = sum(len(c.get("edges", [])) for c in comp2)
    edge_sim = min(edges1, edges2) / max(edges1, edges2, 1)

    # Weighted average
    return 0.4 * count_sim + 0.3 * type_sim + 0.3 * edge_sim


if __name__ == "__main__":
    # Example usage
    from fetch_dataset import load_mbpp_task, list_available_tasks
    from pathlib import Path

    data_dir = Path(__file__).parent / "data"
    task_ids = list_available_tasks(data_dir)

    if task_ids:
        task = load_mbpp_task(task_ids[0], data_dir)
        variants = generate_variants(task)

        print(f"Generated {len(variants)} variants for task {task.task_id}:")
        for v in variants:
            print(f"  - {v.variant_id}: {v.variant_type} ({v.transformations_applied})")
    else:
        print("No tasks found. Run fetch_dataset.py first.")
