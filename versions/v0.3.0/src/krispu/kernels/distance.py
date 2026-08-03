"""Shared normalized-coordinate distance calculations for ARD kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_points(values: ArrayLike, name: str) -> NDArray[np.float64]:
    points = np.asarray(values, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] == 0 or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be a finite two-dimensional array.")
    return points


def scaled_distance_parts(
    first: ArrayLike,
    second: ArrayLike | None,
    length_scale: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return scaled distance and per-axis squared contributions."""

    left = validate_points(first, "X")
    right = left if second is None else validate_points(second, "Y")
    scales = np.asarray(length_scale, dtype=float).reshape(-1)
    if scales.shape != (left.shape[1],) or right.shape[1] != left.shape[1]:
        raise ValueError("length_scale and points must have matching dimensions.")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("length_scale must contain finite positive values.")
    differences = (left[:, None, :] - right[None, :, :]) / scales
    contributions = differences**2
    return np.sqrt(np.sum(contributions, axis=2)), contributions
