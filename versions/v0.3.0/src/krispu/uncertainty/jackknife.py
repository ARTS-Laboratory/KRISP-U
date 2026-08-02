"""LOO field sensitivity and residual diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def loo_field_sensitivity(
    field_means: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(mean, S_LOO)`` for a ``(reference, loo)`` matrix."""

    values = np.asarray(field_means, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("field_means must have shape (n_reference, n_loo) with n_loo > 0.")
    if not np.all(np.isfinite(values)):
        raise ValueError("field_means must contain only finite values.")
    mean = np.mean(values, axis=1)
    n = values.shape[1]
    spread = np.sqrt(np.maximum((n - 1.0) / n * np.sum((values - mean[:, None]) ** 2, axis=1), 0.0))
    return mean, spread


def jackknife_std(field_means: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compatibility alias for :func:`loo_field_sensitivity`."""

    return loo_field_sensitivity(field_means)


def loo_calibration_factor(standardized_residuals: ArrayLike) -> float:
    """Return ``sqrt(median(z_i**2))`` without hiding invalid residuals."""

    values = np.asarray(standardized_residuals, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("standardized LOO residuals must be non-empty and finite.")
    factor = float(np.sqrt(np.median(values**2)))
    if not np.isfinite(factor):
        raise FloatingPointError("LOO calibration factor is non-finite.")
    return factor
