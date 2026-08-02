"""Canonical KRISP-U acquisition: combined candidate-level uncertainty."""

from __future__ import annotations

from numpy.typing import NDArray

from krispu.results import UncertaintyDiagnostics


def loo_uncertainty_scores(diagnostics: UncertaintyDiagnostics) -> NDArray:
    return diagnostics.combined_std.copy()
