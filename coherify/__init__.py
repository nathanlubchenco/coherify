"""
coherify: Formal Coherence Theory for AI Truth-Seeking

A library implementing philosophical coherence measures (Shogenji, Olsson, Fitelson)
for practical AI applications, particularly hallucination detection and reduction.
"""

__version__ = "0.1.0"

from coherify.core.base import CoherenceResult, Proposition, PropositionSet
from coherify.measures.semantic import SemanticCoherence
from coherify.benchmarks.adapters import QABenchmarkAdapter, get_adapter

__all__ = [
    "CoherenceResult",
    "Proposition", 
    "PropositionSet",
    "SemanticCoherence",
    "QABenchmarkAdapter",
    "get_adapter",
]