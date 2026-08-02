"""Diagnostic raw LOO sensitivity acquisition."""

from __future__ import annotations

from numpy.typing import NDArray

from krispu.results import UncertaintyDiagnostics


def raw_loo_sensitivity_scores(diagnostics: UncertaintyDiagnostics) -> NDArray:
    """Return raw LOO field sensitivity for diagnostic comparisons only."""

    return diagnostics.loo_field_sensitivity.copy()
