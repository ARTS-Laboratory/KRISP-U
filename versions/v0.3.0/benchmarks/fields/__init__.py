"""Deterministic two-dimensional audit fields."""

from benchmarks.fields.anisotropic import anisotropic_field
from benchmarks.fields.kernel_fields import (
    periodic_field,
    rough_correlated_field,
    rough_multiscale_field,
    trend_plus_local_field,
)
from benchmarks.fields.localized import localized_field
from benchmarks.fields.noisy import noisy_field
from benchmarks.fields.smooth import smooth_field

FIELD_FACTORIES = {
    "smooth": smooth_field,
    "localized": localized_field,
    "anisotropic": anisotropic_field,
    "noisy_baseline": noisy_field,
    "noisy": noisy_field,
    "rough_correlated": rough_correlated_field,
    "rough_multiscale": rough_multiscale_field,
    "periodic": periodic_field,
    "trend_plus_local": trend_plus_local_field,
}

__all__ = [
    "FIELD_FACTORIES",
    "anisotropic_field",
    "localized_field",
    "noisy_field",
    "periodic_field",
    "rough_correlated_field",
    "rough_multiscale_field",
    "smooth_field",
    "trend_plus_local_field",
]
