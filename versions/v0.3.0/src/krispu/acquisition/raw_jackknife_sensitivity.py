"""Raw buffered-jackknife sensitivity diagnostic."""

from numpy.typing import NDArray

from krispu.results import UncertaintyDiagnostics


def raw_jackknife_sensitivity_scores(diagnostics: UncertaintyDiagnostics) -> NDArray:
    return diagnostics.jackknife_field_sensitivity.copy()
