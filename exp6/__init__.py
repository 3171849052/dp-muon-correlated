"""Experiment 6: local BandInvMF cancellation and clean-p variation diagnostics."""

from .diagnostics import (
    aggregate_window_rows,
    cancellation_score,
    correlation_from_rows,
    spearman_rank_correlation,
    stage_summary,
    window_ranges,
)
from .online_shadow import WindowAccumulatorState, WindowDiagnosticsCollector

__all__ = [
    "WindowAccumulatorState",
    "WindowDiagnosticsCollector",
    "aggregate_window_rows",
    "cancellation_score",
    "correlation_from_rows",
    "spearman_rank_correlation",
    "stage_summary",
    "window_ranges",
]
