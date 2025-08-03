"""
Utilities for clean console output by suppressing non-critical warnings.
"""

import warnings
import logging
from contextlib import contextmanager


def enable_clean_output():
    """
    Enable clean output by suppressing non-critical transformer warnings.

    This suppresses warnings that users cannot control and that don't affect
    functionality, making benchmark output much cleaner and more readable.

    Call this once at the start of your script for cleaner output.
    """
    # Suppress transformer warnings
    warnings.filterwarnings(
        "ignore", category=FutureWarning, message=".*encoder_attention_mask.*"
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*return_all_scores.*"
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*Some weights.*were not initialized.*"
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*You should probably TRAIN this model.*",
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*Device set to use.*"
    )

    # Reduce transformers logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


@contextmanager
def clean_output():
    """
    Context manager for temporarily clean output.

    Usage:
        with clean_output():
            # Run code with suppressed warnings
            result = measure.compute(prop_set)
    """
    # Store original warning filters
    original_filters = warnings.filters.copy()

    # Store original logging levels
    transformers_logger = logging.getLogger("transformers")
    sentence_transformers_logger = logging.getLogger("sentence_transformers")

    original_transformers_level = transformers_logger.level
    original_sentence_transformers_level = sentence_transformers_logger.level

    try:
        # Apply clean output settings
        enable_clean_output()
        yield
    finally:
        # Restore original settings
        warnings.filters = original_filters
        transformers_logger.setLevel(original_transformers_level)
        sentence_transformers_logger.setLevel(original_sentence_transformers_level)


def print_clean_banner():
    """Print a banner indicating clean output mode is enabled."""
    print("🧹 Clean output mode enabled - transformer warnings suppressed")


# Convenience function to enable clean output with user notification
def enable_clean_mode(show_banner: bool = True):
    """
    Enable clean output mode with optional banner.

    Args:
        show_banner: Whether to show a notification that clean mode is enabled
    """
    enable_clean_output()
    if show_banner:
        print_clean_banner()
