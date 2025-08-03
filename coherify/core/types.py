"""
Type definitions and protocols for coherify.
"""

from typing import Protocol, List, Union
import numpy as np
from numpy.typing import NDArray


# Type aliases
ScoreType = float
EmbeddingType = NDArray[np.float32]
SimilarityMatrix = NDArray[np.float32]


class Encoder(Protocol):
    """Protocol for text encoding models."""

    def encode(
        self, texts: Union[str, List[str]], **kwargs
    ) -> Union[EmbeddingType, List[EmbeddingType]]:
        """Encode text(s) into embeddings."""
        ...


class NLIModel(Protocol):
    """Protocol for Natural Language Inference models."""

    def predict(self, premise: str, hypothesis: str) -> str:
        """Predict entailment relationship between premise and hypothesis.

        Returns one of: 'entailment', 'contradiction', 'neutral'
        """
        ...


class SimilarityFunction(Protocol):
    """Protocol for similarity computation functions."""

    def __call__(self, embeddings1: EmbeddingType, embeddings2: EmbeddingType) -> float:
        """Compute similarity between two embeddings."""
        ...
