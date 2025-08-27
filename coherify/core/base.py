"""
Base classes and interfaces for coherence measurement.
Designed to work seamlessly with common NLP benchmark patterns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Proposition:
    """A single proposition with optional probability estimate."""

    text: str
    probability: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PropositionSet:
    """A set of propositions to evaluate for coherence."""

    propositions: List[Proposition]
    context: Optional[str] = None  # For conditional coherence
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def from_qa_pair(
        cls, question: str, answer: str, answer_segments: Optional[List[str]] = None
    ) -> "PropositionSet":
        """Create from QA benchmark format."""
        if answer_segments is None:
            # Simple sentence segmentation
            answer_segments = [s.strip() for s in answer.split(".") if s.strip()]

        props = [Proposition(text=seg) for seg in answer_segments]
        return cls(propositions=props, context=question)

    @classmethod
    def from_multi_answer(cls, question: str, answers: List[str]) -> "PropositionSet":
        """Create from multiple answers to same question (e.g., SelfCheckGPT)."""
        props = [Proposition(text=ans) for ans in answers]
        return cls(propositions=props, context=question)

    def __len__(self) -> int:
        return len(self.propositions)

    def __iter__(self):
        return iter(self.propositions)


@dataclass
class CoherenceResult:
    """Result of coherence measurement."""

    score: float
    measure_name: str
    details: Dict[str, Any]
    computation_time: float

    def exceeds_threshold(self, threshold: float) -> bool:
        """Check if coherence exceeds a threshold."""
        return self.score > threshold

    def __str__(self) -> str:
        return f"{self.measure_name}: {self.score:.3f} (computed in {self.computation_time:.3f}s)"


class CoherenceMeasure(ABC):
    """Abstract base for coherence measures."""

    @abstractmethod
    def compute(self, prop_set: PropositionSet) -> CoherenceResult:
        """Compute coherence for a set of propositions."""

    def compute_pairwise(self, prop1: Proposition, prop2: Proposition) -> float:
        """Compute pairwise coherence between two propositions."""
        prop_set = PropositionSet(propositions=[prop1, prop2])
        result = self.compute(prop_set)
        return result.score

    def batch_compute(self, prop_sets: List[PropositionSet]) -> List[CoherenceResult]:
        """Compute coherence for multiple sets efficiently."""
        return [self.compute(ps) for ps in prop_sets]


class ProbabilityEstimator(ABC):
    """Abstract base for probability estimation."""

    @abstractmethod
    def estimate_probability(
        self, proposition: Proposition, context: Optional[str] = None
    ) -> float:
        """Estimate probability of a single proposition."""

    @abstractmethod
    def estimate_joint_probability(
        self, propositions: List[Proposition], context: Optional[str] = None
    ) -> float:
        """Estimate joint probability of multiple propositions."""
