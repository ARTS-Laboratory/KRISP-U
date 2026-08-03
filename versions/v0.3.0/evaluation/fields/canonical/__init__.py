"""Canonical benchmark fields."""

from evaluation.fields.canonical.anisotropic import anisotropic_field
from evaluation.fields.canonical.localized import localized_field
from evaluation.fields.canonical.noisy import noisy_field
from evaluation.fields.canonical.smooth import smooth_field

__all__ = ["anisotropic_field", "localized_field", "noisy_field", "smooth_field"]
