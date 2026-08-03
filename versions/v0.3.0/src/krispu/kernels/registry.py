"""Finite registry of single global anisotropic spatial covariance families."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.gaussian_process.kernels import WhiteKernel

from krispu.kernels.families import (
    ExponentialARD,
    GaussianARD,
    Matern32ARD,
    Matern52ARD,
    RationalQuadraticARD,
    SphericalARD,
    WendlandC2ARD,
)

Builder = Callable[[int, bool], Any]


@dataclass(frozen=True)
class KernelDefinition:
    kernel_id: str
    display_name: str
    builder: Builder
    number_of_hyperparameters: int
    supported_dimensions: tuple[int, ...] | str
    profile_tags: tuple[str, ...]
    default_bounds: dict[str, tuple[float, float]]
    supports_measurement_noise: bool
    spatial_components: int = 1


def _bounds(value: tuple[float, float], optimize: bool) -> tuple[float, float] | str:
    return value if optimize else "fixed"


def _family(
    family: type[Any],
    dimension: int,
    optimize: bool,
    *,
    rational_quadratic: bool = False,
) -> Any:
    if dimension < 1:
        raise ValueError("dimension must be positive.")
    if family in {SphericalARD, WendlandC2ARD} and dimension > 3:
        raise ValueError(f"{family.__name__} supports dimensions one through three.")
    scale_bounds: Any = (
        np.broadcast_to(np.asarray((0.02, 2.0), dtype=float), (dimension, 2)).copy()
        if optimize
        else "fixed"
    )
    kwargs: dict[str, Any] = {
        "amplitude": 1.0,
        "amplitude_bounds": _bounds((1e-3, 5.0), optimize),
        "length_scale": np.full(dimension, 0.25, dtype=float),
        "length_scale_bounds": scale_bounds,
    }
    if rational_quadratic:
        kwargs.update({"alpha": 1.0, "alpha_bounds": _bounds((1e-3, 100.0), optimize)})
    return family(**kwargs)


def _gaussian(dimension: int, optimize: bool) -> Any:
    return _family(GaussianARD, dimension, optimize)


def _exponential(dimension: int, optimize: bool) -> Any:
    return _family(ExponentialARD, dimension, optimize)


def _spherical(dimension: int, optimize: bool) -> Any:
    return _family(SphericalARD, dimension, optimize)


def _matern32(dimension: int, optimize: bool) -> Any:
    return _family(Matern32ARD, dimension, optimize)


def _matern52(dimension: int, optimize: bool) -> Any:
    return _family(Matern52ARD, dimension, optimize)


def _rational_quadratic(dimension: int, optimize: bool) -> Any:
    return _family(RationalQuadraticARD, dimension, optimize, rational_quadratic=True)


def _wendland(dimension: int, optimize: bool) -> Any:
    return _family(WendlandC2ARD, dimension, optimize)


KERNEL_REGISTRY: dict[str, KernelDefinition] = {
    "gaussian_ard": KernelDefinition(
        "gaussian_ard", "Gaussian ARD", _gaussian, 1, "continuous", ("smooth",),
        {"length_scale": (0.02, 2.0), "amplitude": (1e-3, 5.0)}, True,
    ),
    "exponential_ard": KernelDefinition(
        "exponential_ard", "Exponential ARD", _exponential, 1, "continuous", ("rough",),
        {"length_scale": (0.02, 2.0), "amplitude": (1e-3, 5.0)}, True,
    ),
    "spherical_ard": KernelDefinition(
        "spherical_ard", "Spherical ARD", _spherical, 1, (1, 2, 3), ("compact",),
        {"length_scale": (0.02, 2.0), "amplitude": (1e-3, 5.0)}, True,
    ),
    "matern_32_ard": KernelDefinition(
        "matern_32_ard", "Matérn-3/2 ARD", _matern32, 1, "continuous", ("rough",),
        {"length_scale": (0.02, 2.0), "amplitude": (1e-3, 5.0)}, True,
    ),
    "matern_52_ard": KernelDefinition(
        "matern_52_ard", "Matérn-5/2 ARD", _matern52, 1, "continuous", ("smooth",),
        {"length_scale": (0.02, 2.0), "amplitude": (1e-3, 5.0)}, True,
    ),
    "rational_quadratic_ard": KernelDefinition(
        "rational_quadratic_ard", "Rational Quadratic ARD", _rational_quadratic, 2,
        "continuous", ("multiscale",),
        {"length_scale": (0.02, 2.0), "alpha": (1e-3, 100.0)}, True,
    ),
    "wendland_c2_ard": KernelDefinition(
        "wendland_c2_ard", "Wendland C2 ARD", _wendland, 1, (1, 2, 3), ("compact",),
        {"length_scale": (0.02, 2.0), "amplitude": (1e-3, 5.0)}, True,
    ),
}

CANDIDATE_SETS: dict[str, tuple[str, ...]] = {"standard": tuple(KERNEL_REGISTRY)}


def registered_kernel_ids() -> tuple[str, ...]:
    return tuple(KERNEL_REGISTRY)


def get_kernel_definition(kernel_id: str) -> KernelDefinition:
    try:
        return KERNEL_REGISTRY[kernel_id]
    except KeyError as exc:
        raise ValueError(f"Unknown registered kernel: {kernel_id}") from exc


def candidate_ids(candidate_set: str = "standard") -> tuple[str, ...]:
    try:
        return CANDIDATE_SETS[candidate_set]
    except KeyError as exc:
        raise ValueError(f"Unknown registered candidate set: {candidate_set}") from exc


def add_observation_noise(kernel: Any, *, initial: float, bounds: tuple[float, float], optimize: bool) -> Any:
    """Add only a separate diagonal observation-noise term when requested."""

    return kernel + WhiteKernel(
        noise_level=initial,
        noise_level_bounds=bounds if optimize else "fixed",
    )
