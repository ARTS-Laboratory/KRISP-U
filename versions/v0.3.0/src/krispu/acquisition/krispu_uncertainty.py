"""Canonical support-adjusted KRISP-U acquisition."""

from __future__ import annotations

from numpy.typing import NDArray

from krispu.results import UncertaintyDiagnostics


def krispu_uncertainty_scores(diagnostics: UncertaintyDiagnostics) -> NDArray:
    """Return the canonical buffered-jackknife ``S * sqrt(C)`` field."""

    return diagnostics.krispu_uncertainty.copy()
