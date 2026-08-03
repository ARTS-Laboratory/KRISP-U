"""Gaussian-process sampled fields for covariance-recovery experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from evaluation.fields.canonical.smooth import AuditField
from krispu.domains import ContinuousDomain
from krispu.kernels.builders import build_named_kernel


def sampled_gp_field(
    family: str = "gaussian_ard",
    amplitude: float = 1.0,
    length_scales: tuple[float, float] = (0.25, 0.25),
    nugget: float = 1.0e-6,
    seed: int = 0,
    *,
    rotation_degrees: float = 0.0,
    axis_scale: tuple[float, float] = (1.0, 1.0),
) -> AuditField:
    """Sample one GP on a locked grid and expose it as a deterministic field.

    The interpolation is only a storage representation of one finite GP draw;
    metadata retains the exact covariance parameters used to generate it.
    """

    if len(length_scales) != 2 or any(value <= 0 for value in length_scales):
        raise ValueError("length_scales must contain two positive values")
    if any(value <= 0 for value in axis_scale):
        raise ValueError("axis_scale must contain positive values")
    domain = ContinuousDomain([[-1.0, 1.0], [-1.0, 1.0]], names=("x", "y"))
    grid_axis = np.linspace(-1.0, 1.0, 28)
    mesh = np.meshgrid(grid_axis, grid_axis, indexing="xy")
    grid = np.column_stack([mesh[0].ravel(), mesh[1].ravel()])
    transformed = _transform(grid, rotation_degrees, axis_scale)
    kernel = build_named_kernel(family, 2, optimize_hyperparameters=False)
    kernel = kernel.clone_with_theta(kernel.theta + 0.0)
    kernel.length_scale = np.asarray(length_scales, dtype=float)
    kernel.amplitude = float(amplitude)
    covariance = kernel(transformed, transformed) + float(nugget) * np.eye(len(grid))
    covariance += 1.0e-10 * np.eye(len(grid))
    sample = np.random.default_rng(seed).multivariate_normal(np.zeros(len(grid)), covariance)
    interpolator = RegularGridInterpolator(
        (grid_axis, grid_axis), sample.reshape(28, 28), bounds_error=True
    )

    def evaluate(values: Any) -> np.ndarray:
        points = np.asarray(values, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        return np.asarray(interpolator(points), dtype=float)

    return AuditField(
        name=f"gp_{family}",
        domain=domain,
        evaluate=evaluate,
        recommended_plot_limits=(float(np.min(sample)), float(np.max(sample))),
        metadata={
            "description": "One finite Gaussian-process draw",
            "field_category": "synthetic_covariance_recovery",
            "true_kernel": {
                "family": family,
                "amplitude": float(amplitude),
                "ard_length_scales": [float(value) for value in length_scales],
                "nugget": float(nugget),
                "seed": int(seed),
                "rotation_degrees": float(rotation_degrees),
                "axis_scale": [float(value) for value in axis_scale],
            },
        },
    )


def rotated_anisotropic_gp_field(seed: int = 0) -> AuditField:
    return sampled_gp_field(
        "matern_32_ard",
        length_scales=(0.10, 0.45),
        nugget=1.0e-5,
        seed=seed,
        rotation_degrees=35.0,
    )


def axis_rescaled_anisotropic_gp_field(seed: int = 0) -> AuditField:
    return sampled_gp_field(
        "gaussian_ard",
        length_scales=(0.12, 0.40),
        nugget=1.0e-5,
        seed=seed,
        axis_scale=(1.6, 0.7),
    )


def _candidate_gp_field(family: str, seed: int = 0) -> AuditField:
    return sampled_gp_field(
        family,
        length_scales=(0.18, 0.42),
        nugget=1.0e-5,
        seed=seed,
    )


def gaussian_gp_field(seed: int = 0) -> AuditField:
    return _candidate_gp_field("gaussian_ard", seed)


def exponential_gp_field(seed: int = 0) -> AuditField:
    return _candidate_gp_field("exponential_ard", seed)


def spherical_gp_field(seed: int = 0) -> AuditField:
    return _candidate_gp_field("spherical_ard", seed)


def matern32_gp_field(seed: int = 0) -> AuditField:
    return _candidate_gp_field("matern_32_ard", seed)


def matern52_gp_field(seed: int = 0) -> AuditField:
    return _candidate_gp_field("matern_52_ard", seed)


def rational_quadratic_gp_field(seed: int = 0) -> AuditField:
    return _candidate_gp_field("rational_quadratic_ard", seed)


def wendland_gp_field(seed: int = 0) -> AuditField:
    return _candidate_gp_field("wendland_c2_ard", seed)


__all__ = [
    "axis_rescaled_anisotropic_gp_field",
    "exponential_gp_field",
    "gaussian_gp_field",
    "matern32_gp_field",
    "matern52_gp_field",
    "rational_quadratic_gp_field",
    "rotated_anisotropic_gp_field",
    "sampled_gp_field",
    "spherical_gp_field",
    "wendland_gp_field",
]


def _transform(
    points: np.ndarray,
    rotation_degrees: float,
    axis_scale: tuple[float, float],
) -> np.ndarray:
    angle = np.deg2rad(rotation_degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (points @ rotation) / np.asarray(axis_scale, dtype=float)
