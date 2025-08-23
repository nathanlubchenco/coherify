"""
User interface components for Coherify benchmark results visualization.

Provides simple web UI for viewing comprehensive benchmark reports.
"""

from .result_viewer import ResultViewer, start_result_server

__all__ = [
    "ResultViewer",
    "start_result_server",
]