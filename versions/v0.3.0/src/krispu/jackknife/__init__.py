"""Buffered-jackknife planning and field sensitivity."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from krispu.jackknife.plan import BufferedJackknifePlan, build_buffered_jackknife_plan


def jackknife_field_sensitivity(
    field_means: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = np.asarray(field_means, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("field_means must be finite with shape (n_reference, n_folds).")
    mean = np.mean(values, axis=1)
    n = values.shape[1]
    sensitivity = np.sqrt(
        np.maximum((n - 1.0) / n * np.sum((values - mean[:, None]) ** 2, axis=1), 0.0)
    )
    return mean, sensitivity


def jackknife_calibration_factor(standardized_residuals: ArrayLike) -> float:
    values = np.asarray(standardized_residuals, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("standardized jackknife residuals must be non-empty and finite.")
    return float(np.sqrt(np.median(values**2)))


__all__ = [
    "BufferedJackknifePlan",
    "build_buffered_jackknife_plan",
    "jackknife_calibration_factor",
    "jackknife_field_sensitivity",
]
