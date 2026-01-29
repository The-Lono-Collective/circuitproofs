"""
MBPP-Lean Benchmark Integration

Provides tools for validating circuit extraction against
MBPP problems with formal Lean specifications.
"""

from benchmarks.verina.fetch_dataset import (
    fetch_verina_dataset,
    load_mbpp_task,
    list_available_tasks,
    MBPPTask,
)

__all__ = [
    "fetch_verina_dataset",
    "load_mbpp_task",
    "list_available_tasks",
    "MBPPTask",
]
