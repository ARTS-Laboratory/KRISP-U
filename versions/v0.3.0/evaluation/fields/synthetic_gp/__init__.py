"""Synthetic Gaussian-process benchmark fields."""

from evaluation.fields.synthetic_gp.kernel_fields import (
    periodic_field,
    rough_correlated_field,
    rough_multiscale_field,
    trend_plus_local_field,
)

__all__ = [
    "periodic_field",
    "rough_correlated_field",
    "rough_multiscale_field",
    "trend_plus_local_field",
]
"""Fields sampled from known Gaussian-process covariance families."""

from evaluation.fields.synthetic_gp.sampled import (
    axis_rescaled_anisotropic_gp_field,
    rotated_anisotropic_gp_field,
    sampled_gp_field,
)

__all__ = ["axis_rescaled_anisotropic_gp_field", "rotated_anisotropic_gp_field", "sampled_gp_field"]
