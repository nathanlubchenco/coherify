"""
Utilities for working with transformers library and suppressing common warnings.
"""

import warnings
from contextlib import contextmanager
from typing import Any, Callable


@contextmanager
def suppress_transformer_warnings():
    """
    Context manager to suppress common transformer model warnings.

    This suppresses specific FutureWarnings about deprecated parameters
    that are not in our control but come from the transformers library.
    """
    with warnings.catch_warnings():
        # Suppress the specific encoder_attention_mask deprecation warning
        warnings.filterwarnings(
            "ignore", category=FutureWarning, message=".*encoder_attention_mask.*"
        )

        # Suppress return_all_scores deprecation warning
        warnings.filterwarnings(
            "ignore", category=UserWarning, message=".*return_all_scores.*"
        )

        # Suppress other common transformer warnings that users can't control
        warnings.filterwarnings(
            "ignore", category=FutureWarning, message=".*attention_mask.*"
        )

        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Some weights.*were not initialized.*",
        )

        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*You should probably TRAIN this model.*",
        )

        # Suppress MPS device warnings
        warnings.filterwarnings(
            "ignore", category=UserWarning, message=".*Device set to use.*"
        )

        # Suppress informational messages that appear during model loading
        import logging

        # Temporarily reduce transformers logging level
        transformers_logger = logging.getLogger("transformers")
        original_level = transformers_logger.level
        transformers_logger.setLevel(logging.ERROR)

        # Store original level to restore later
        if not hasattr(warnings, "_transformers_original_level"):
            warnings._transformers_original_level = original_level

        yield


def safe_pipeline_call(pipeline_func: Callable, *args, **kwargs) -> Any:
    """
    Safely call a transformers pipeline function with warning suppression.

    Args:
        pipeline_func: The pipeline function to call
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of the pipeline function call
    """
    with suppress_transformer_warnings():
        return pipeline_func(*args, **kwargs)


def create_pipeline_with_suppressed_warnings(task: str, model: str, **kwargs):
    """
    Create a transformers pipeline with suppressed warnings.

    Args:
        task: The pipeline task (e.g., "text-classification")
        model: The model name or path
        **kwargs: Additional arguments for the pipeline

    Returns:
        Configured pipeline
    """
    from transformers import pipeline

    with suppress_transformer_warnings():
        return pipeline(task, model=model, **kwargs)
