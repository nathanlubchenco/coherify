"""
Evaluation orchestration and aggregation.

This module provides higher-level evaluation patterns including:
- Majority voting across multiple runs
- K-run orchestration and aggregation
- Confidence-based result combination
"""

from .majority_voting import MajorityVotingEvaluator
from .k_run import KRunBenchmarkEvaluator

__all__ = [
    "MajorityVotingEvaluator", 
    "KRunBenchmarkEvaluator"
]