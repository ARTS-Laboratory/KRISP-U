"""Deterministic two-dimensional audit fields."""

from benchmarks.fields.anisotropic import anisotropic_field
from benchmarks.fields.localized import localized_field
from benchmarks.fields.smooth import smooth_field

FIELD_FACTORIES = {
    "smooth": smooth_field,
    "localized": localized_field,
    "anisotropic": anisotropic_field,
}

__all__ = ["FIELD_FACTORIES", "anisotropic_field", "localized_field", "smooth_field"]
