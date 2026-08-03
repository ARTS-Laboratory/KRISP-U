"""Support-adjusted buffered-jackknife acquisition field."""

from numpy.typing import NDArray

from krispu.results import UncertaintyDiagnostics


def jackknife_uncertainty_scores(diagnostics: UncertaintyDiagnostics) -> NDArray:
    return diagnostics.krispu_uncertainty.copy()
