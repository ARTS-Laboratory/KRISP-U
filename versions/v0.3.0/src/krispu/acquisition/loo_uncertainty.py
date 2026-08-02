"""Canonical KRISP-U acquisition: candidate-level LOO field uncertainty."""

from __future__ import annotations

from numpy.typing import NDArray

from krispu.results import UncertaintyDiagnostics


def loo_uncertainty_scores(diagnostics: UncertaintyDiagnostics) -> NDArray:
    return diagnostics.loo_field_uncertainty.copy()
