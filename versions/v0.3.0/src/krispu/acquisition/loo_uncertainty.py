"""Compatibility import for the former LOO acquisition module."""

from __future__ import annotations

from numpy.typing import NDArray

from krispu.acquisition.krispu_uncertainty import krispu_uncertainty_scores
from krispu.results import UncertaintyDiagnostics


def loo_uncertainty_scores(diagnostics: UncertaintyDiagnostics) -> NDArray:
    """Compatibility alias for the canonical support-adjusted score."""

    return krispu_uncertainty_scores(diagnostics)
