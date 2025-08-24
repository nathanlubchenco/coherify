"""
Official benchmark evaluation implementations.

This module contains faithful reproductions of original benchmark evaluation methods.
These MUST match the official implementations exactly to establish proper baselines.

CRITICAL: Never skip official evaluation. Always establish baselines first.
"""

try:
    from .truthfulqa_official import TruthfulQAOfficialEvaluator
    HAS_TRUTHFULQA = True
except ImportError:
    HAS_TRUTHFULQA = False

try:
    from .fever_official import FEVEROfficialEvaluator
    HAS_FEVER = True
except ImportError:
    HAS_FEVER = False

try:
    from .selfcheckgpt_official import SelfCheckGPTOfficialEvaluator
    HAS_SELFCHECKGPT = True
except ImportError:
    HAS_SELFCHECKGPT = False

__all__ = []

if HAS_TRUTHFULQA:
    __all__.append("TruthfulQAOfficialEvaluator")

if HAS_FEVER:
    __all__.append("FEVEROfficialEvaluator")

if HAS_SELFCHECKGPT:
    __all__.append("SelfCheckGPTOfficialEvaluator")