"""Explicit posterior-standard-deviation baseline."""

from __future__ import annotations

from numpy.typing import NDArray

from krispu.results import UncertaintyDiagnostics


def posterior_std_scores(diagnostics: UncertaintyDiagnostics) -> NDArray:
    return diagnostics.posterior_std.copy()
