"""Validated builders for the finite single-family kernel registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from krispu.kernels.registry import add_observation_noise, get_kernel_definition
from krispu.kernels.specification import as_float_pair, as_length_scale

_TYPE_TO_ID = {
    "gaussian": "gaussian_ard",
    "gaussian_ard": "gaussian_ard",
    "exponential": "exponential_ard",
    "exponential_ard": "exponential_ard",
    "spherical": "spherical_ard",
    "spherical_ard": "spherical_ard",
    "matern_32": "matern_32_ard",
    "matern_32_ard": "matern_32_ard",
    "matern_52": "matern_52_ard",
    "matern_52_ard": "matern_52_ard",
    "rational_quadratic": "rational_quadratic_ard",
    "rational_quadratic_ard": "rational_quadratic_ard",
    "wendland_c2": "wendland_c2_ard",
    "wendland_c2_ard": "wendland_c2_ard",
}


def build_kernel_from_spec(
    specification: Mapping[str, Any],
    dimension: int,
    optimize_hyperparameters: bool = True,
) -> Any:
    """Build one explicit ARD family; expression trees are intentionally rejected."""

    if dimension < 1:
        raise ValueError("dimension must be positive.")
    spec = dict(specification)
    kind = str(spec.get("type", "")).lower().replace("-", "_")
    if kind in {"additive", "sum", "multiplicative", "product"} or "components" in spec:
        raise ValueError("Manual kernel specifications may contain exactly one spatial family.")
    try:
        kernel_id = _TYPE_TO_ID[kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported explicit kernel type: {specification.get('type')!r}") from exc
    definition = get_kernel_definition(kernel_id)
    length_scale = np.asarray(
        as_length_scale(spec.get("length_scale_initial", 0.25), dimension, "length_scale_initial"),
        dtype=float,
    )
    bounds_value = spec.get("length_scale_bounds", [0.02, 2.0])
    minimum, maximum = _expanded_bounds(bounds_value, dimension, "length_scale_bounds")
    kwargs: dict[str, Any] = {
        "amplitude": float(spec.get("amplitude_initial", 1.0)),
        "amplitude_bounds": _bounds(
            as_float_pair(spec.get("amplitude_bounds", [1e-3, 5.0]), "amplitude_bounds"),
            optimize_hyperparameters,
        ),
        "length_scale": length_scale,
        "length_scale_bounds": (
            np.column_stack((np.full(dimension, minimum), np.full(dimension, maximum)))
            if optimize_hyperparameters
            else "fixed"
        ),
    }
    if kernel_id == "rational_quadratic_ard":
        kwargs["alpha"] = float(spec.get("alpha_initial", 1.0))
        kwargs["alpha_bounds"] = _bounds(
            as_float_pair(spec.get("alpha_bounds", [1e-3, 100.0]), "alpha_bounds"),
            optimize_hyperparameters,
        )
    kernel = _construct(definition, kwargs)
    noise = spec.get("observation_noise")
    if noise is not None:
        if not isinstance(noise, Mapping):
            raise ValueError("observation_noise must be a mapping.")
        if bool(noise.get("enabled", False)):
            kernel = add_observation_noise(
                kernel,
                initial=float(noise.get("initial", 1e-6)),
                bounds=as_float_pair(noise.get("bounds", [1e-10, 1.0]), "observation_noise.bounds"),
                optimize=optimize_hyperparameters,
            )
    return kernel


def build_named_kernel(kernel_id: str, dimension: int, optimize_hyperparameters: bool = True) -> Any:
    definition = get_kernel_definition(kernel_id)
    return definition.builder(dimension, optimize_hyperparameters)


def _construct(definition: Any, kwargs: dict[str, Any]) -> Any:
    family = definition.builder(1, True).__class__
    if definition.kernel_id == "rational_quadratic_ard":
        from krispu.kernels.families.rational_quadratic import RationalQuadraticARD

        family = RationalQuadraticARD
    return family(**kwargs)


def _expanded_bounds(value: Any, dimension: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(value, dtype=float)
    if array.shape == (2,):
        lower = np.full(dimension, array[0], dtype=float)
        upper = np.full(dimension, array[1], dtype=float)
    elif array.shape == (dimension, 2):
        lower, upper = array[:, 0], array[:, 1]
    else:
        raise ValueError(f"{name} must be [minimum, maximum] or one pair per dimension.")
    if not np.all(np.isfinite(array)) or np.any(lower <= 0) or np.any(lower >= upper):
        raise ValueError(f"{name} must contain increasing positive bounds.")
    return lower, upper


def _bounds(bounds: tuple[float, float], optimize: bool) -> tuple[float, float] | str:
    return bounds if optimize else "fixed"
