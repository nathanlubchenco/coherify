"""
Temperature selection strategies for diverse response generation.

This module implements various temperature selection strategies to optimize
response diversity and quality for different types of questions.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class TemperatureProfile:
    """Profile for temperature selection strategies."""

    base_temp: float
    range_min: float
    range_max: float
    distribution: str = "uniform"  # uniform, exponential, gaussian

    def sample(self, k: int) -> List[float]:
        """Sample K temperatures according to the distribution."""
        if self.distribution == "uniform":
            return list(np.linspace(self.range_min, self.range_max, k))
        elif self.distribution == "exponential":
            # Exponential spacing for more diversity
            log_temps = np.linspace(
                np.log(max(0.01, self.range_min)), np.log(self.range_max), k
            )
            return list(np.exp(log_temps))
        elif self.distribution == "gaussian":
            # Centered around base_temp with variation
            return list(
                np.random.normal(self.base_temp, 0.2, k).clip(
                    self.range_min, self.range_max
                )
            )
        else:
            # Default to uniform
            return list(np.linspace(self.range_min, self.range_max, k))


class AdaptiveTemperatureSelector:
    """
    Selects temperatures adaptively based on question characteristics.

    Implements the recommendations from OpenAI documentation:
    - Lower temperatures (0.1-0.5) for factual/deterministic questions
    - Medium temperatures (0.5-1.0) for balanced creativity
    - Higher temperatures (1.0-2.0) for creative/open-ended questions
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize adaptive temperature selector.

        Args:
            config: Optional configuration for temperature ranges
        """
        self.config = config or {}

        # Default profiles based on OpenAI recommendations
        self.profiles = {
            "factual": TemperatureProfile(
                base_temp=0.3, range_min=0.1, range_max=0.5, distribution="uniform"
            ),
            "balanced": TemperatureProfile(
                base_temp=0.7, range_min=0.3, range_max=1.0, distribution="exponential"
            ),
            "creative": TemperatureProfile(
                base_temp=1.2, range_min=0.7, range_max=1.8, distribution="gaussian"
            ),
            "exploratory": TemperatureProfile(
                base_temp=1.5, range_min=1.0, range_max=2.0, distribution="exponential"
            ),
        }

        # Update with config if provided
        if config and "temperature_profiles" in config:
            self.profiles.update(config["temperature_profiles"])

    def classify_question(self, question: str) -> str:
        """
        Classify question type to determine temperature profile.

        Args:
            question: The input question

        Returns:
            Question type: 'factual', 'balanced', 'creative', or 'exploratory'
        """
        question_lower = question.lower()

        # Factual indicators
        factual_keywords = [
            "what is",
            "when did",
            "who is",
            "where is",
            "how many",
            "define",
            "list",
            "name",
            "which",
            "calculate",
            "true or false",
        ]
        if any(keyword in question_lower for keyword in factual_keywords):
            return "factual"

        # Creative indicators
        creative_keywords = [
            "imagine",
            "create",
            "design",
            "story",
            "poem",
            "creative",
            "what if",
            "suppose",
            "hypothetical",
            "fiction",
        ]
        if any(keyword in question_lower for keyword in creative_keywords):
            return "creative"

        # Exploratory indicators
        exploratory_keywords = [
            "explore",
            "discuss",
            "analyze",
            "compare",
            "contrast",
            "opinion",
            "perspective",
            "elaborate",
            "explain why",
        ]
        if any(keyword in question_lower for keyword in exploratory_keywords):
            return "exploratory"

        # Default to balanced
        return "balanced"

    def select_temperatures(
        self, question: str, k: int = 5, difficulty: Optional[str] = None
    ) -> List[float]:
        """
        Select K temperatures for a given question.

        Args:
            question: The input question
            k: Number of temperatures to select
            difficulty: Optional difficulty override ('easy', 'medium', 'hard')

        Returns:
            List of K temperature values
        """
        # Use difficulty-based selection if provided
        if difficulty:
            if difficulty == "easy":
                profile = TemperatureProfile(0.3, 0.1, 0.5, "uniform")
            elif difficulty == "hard":
                profile = TemperatureProfile(1.0, 0.5, 1.5, "exponential")
            else:  # medium
                profile = TemperatureProfile(0.7, 0.3, 1.0, "uniform")
        else:
            # Classify question and get appropriate profile
            question_type = self.classify_question(question)
            profile = self.profiles[question_type]

        return profile.sample(k)

    def get_temperature_schedule(
        self, questions: List[str], k: int = 5
    ) -> List[List[float]]:
        """
        Get temperature schedules for a batch of questions.

        Args:
            questions: List of questions
            k: Number of responses per question

        Returns:
            List of temperature lists, one per question
        """
        schedules = []
        for question in questions:
            temps = self.select_temperatures(question, k)
            schedules.append(temps)
        return schedules


class ProgressiveTemperatureStrategy:
    """
    Progressive temperature strategy that starts low and increases.

    This is useful for generating a diverse set of responses where
    we want at least one very conservative/factual response and
    progressively more creative ones.
    """

    def __init__(self, start: float = 0.1, end: float = 1.5):
        """
        Initialize progressive strategy.

        Args:
            start: Starting temperature
            end: Ending temperature
        """
        self.start = start
        self.end = end

    def get_temperatures(self, k: int) -> List[float]:
        """Get K temperatures in progressive order."""
        if k == 1:
            return [(self.start + self.end) / 2]
        return list(np.linspace(self.start, self.end, k))


class EntropyBasedTemperatureStrategy:
    """
    Adjust temperature based on response entropy/diversity.

    If initial responses are too similar, increase temperature.
    If responses are too random, decrease temperature.
    """

    def __init__(self, target_entropy: float = 0.7):
        """
        Initialize entropy-based strategy.

        Args:
            target_entropy: Target entropy level (0-1)
        """
        self.target_entropy = target_entropy
        self.history = []

    def compute_entropy(self, responses: List[str]) -> float:
        """
        Compute entropy of response set.

        Simple word-level entropy calculation.
        """
        from collections import Counter

        all_words = []
        for response in responses:
            words = response.lower().split()
            all_words.extend(words)

        if not all_words:
            return 0.0

        word_counts = Counter(all_words)
        total = len(all_words)

        entropy = 0.0
        for count in word_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * np.log2(prob)

        # Normalize by max possible entropy
        max_entropy = np.log2(len(word_counts)) if word_counts else 1.0
        return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0

    def adjust_temperature(self, current_temp: float, responses: List[str]) -> float:
        """
        Adjust temperature based on response entropy.

        Args:
            current_temp: Current temperature
            responses: Generated responses

        Returns:
            Adjusted temperature
        """
        entropy = self.compute_entropy(responses)
        self.history.append(entropy)

        # Calculate adjustment factor
        entropy_diff = entropy - self.target_entropy

        # Adjust temperature (higher entropy -> lower temp, lower entropy -> higher temp)
        adjustment = -entropy_diff * 0.3  # Scale factor
        new_temp = current_temp + adjustment

        # Clamp to valid range
        return max(0.1, min(2.0, new_temp))


# Convenience function
def get_optimal_temperatures(
    question: str, k: int = 5, strategy: str = "adaptive"
) -> List[float]:
    """
    Get optimal temperatures for a question using specified strategy.

    Args:
        question: The input question
        k: Number of temperatures needed
        strategy: Strategy to use ('adaptive', 'progressive', 'uniform')

    Returns:
        List of K temperature values
    """
    if strategy == "adaptive":
        selector = AdaptiveTemperatureSelector()
        return selector.select_temperatures(question, k)
    elif strategy == "progressive":
        strategy = ProgressiveTemperatureStrategy()
        return strategy.get_temperatures(k)
    else:  # uniform
        return list(np.linspace(0.3, 1.0, k))
