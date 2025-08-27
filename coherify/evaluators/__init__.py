"""
Evaluation orchestration and aggregation.

This module provides higher-level evaluation patterns including:
- Majority voting across multiple runs
- K-run orchestration and aggregation
- Confidence-based result combination
"""

from .k_run import KRunBenchmarkEvaluator
from .majority_voting import MajorityVotingEvaluator

__all__ = ["MajorityVotingEvaluator", "KRunBenchmarkEvaluator"]
