"""Kernel support-deficit calculations for the canonical KRISP-U field."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.gaussian_process.kernels import (
    Exponentiation,
    Kernel,
    Product,
    Sum,
    WhiteKernel,
)

from krispu.surrogates.gpr import GPRSurrogate


def kernel_support_deficit(
    surrogate: GPRSurrogate,
    observation_points_normalized: ArrayLike,
    reference_points_normalized: ArrayLike,
    epsilon: float = 1e-12,
    condition_number_limit: float = 1e12,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(C, max_kernel_correlation)`` for normalized points.

    The fitted latent-process kernel is used for every covariance involving a
    candidate.  The fitted ``WhiteKernel`` terms and the GPR ``alpha`` term are
    moved to the observation-noise diagonal instead of being included in
    ``k_f(x, x)``.
    """

    if surrogate.model_ is None or surrogate.kernel_ is None:
        raise ValueError("The surrogate must be fitted before support is evaluated.")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be a finite positive scalar.")
    if condition_number_limit <= 1 or not np.isfinite(condition_number_limit):
        raise ValueError("condition_number_limit must be finite and greater than one.")

    observations = _points(observation_points_normalized, "observation_points_normalized")
    references = _points(reference_points_normalized, "reference_points_normalized")
    if observations.shape[1] != references.shape[1]:
        raise ValueError("observation and reference points must have matching dimensions.")

    latent_kernel = _without_white_noise(surrogate.kernel_)
    if latent_kernel is None:
        raise ValueError("The fitted kernel has no latent-process component.")
    latent_train = np.asarray(latent_kernel(observations), dtype=float)
    latent_cross = np.asarray(latent_kernel(references, observations), dtype=float)
    latent_reference = np.asarray(latent_kernel(references), dtype=float)
    _require_finite(latent_train, "latent training covariance")
    _require_finite(latent_cross, "latent cross covariance")
    _require_finite(latent_reference, "latent reference covariance")

    if latent_train.shape != (len(observations), len(observations)):
        raise ValueError("The fitted latent kernel returned an invalid training covariance shape.")
    if latent_cross.shape != (len(references), len(observations)):
        raise ValueError("The fitted latent kernel returned an invalid cross covariance shape.")
    if latent_reference.shape != (len(references), len(references)):
        raise ValueError("The fitted latent kernel returned an invalid reference covariance shape.")

    noise = _observation_noise(surrogate, len(observations))
    covariance = latent_train + np.diag(noise)
    _require_finite(covariance, "training covariance")
    condition_number = float(np.linalg.cond(covariance))
    if not np.isfinite(condition_number) or condition_number > condition_number_limit:
        raise FloatingPointError(
            "The fitted support covariance is non-finite or badly conditioned "
            f"(condition number={condition_number:.3g})."
        )
    try:
        solved = np.linalg.solve(covariance, latent_cross.T)
    except np.linalg.LinAlgError as exc:
        raise FloatingPointError("The fitted support covariance cannot be solved stably.") from exc
    _require_finite(solved, "support covariance solve")

    latent_variance = np.diag(latent_reference).copy()
    explained = np.sum(latent_cross * solved.T, axis=1)
    _require_finite(latent_variance, "latent reference variance")
    _require_finite(explained, "explained latent variance")
    if np.any(latent_variance < -epsilon):
        raise FloatingPointError("The latent kernel produced a negative reference variance.")
    latent_variance = np.maximum(latent_variance, 0.0)
    support_variance = latent_variance - explained
    if np.any(support_variance < -100.0 * epsilon * np.maximum(latent_variance, 1.0)):
        raise FloatingPointError("The calculated support variance is materially negative.")
    support_variance = np.maximum(support_variance, 0.0)
    denominator = np.maximum(latent_variance, epsilon)
    deficit = np.clip(support_variance / denominator, 0.0, 1.0)

    correlations = _kernel_correlations(latent_cross, latent_variance, latent_train)
    _require_finite(deficit, "kernel support deficit")
    _require_finite(correlations, "maximum kernel correlation")
    return deficit, correlations


def _observation_noise(surrogate: GPRSurrogate, count: int) -> NDArray[np.float64]:
    assert surrogate.model_ is not None
    alpha = np.asarray(surrogate.model_.alpha, dtype=float)
    if alpha.ndim == 0:
        noise = np.full(count, float(alpha), dtype=float)
    elif alpha.shape == (count,):
        noise = alpha.copy()
    else:
        raise ValueError("The fitted GPR alpha has an invalid observation-noise shape.")
    noise += _white_noise_level(surrogate.kernel_)
    if not np.all(np.isfinite(noise)) or np.any(noise < 0):
        raise FloatingPointError("The fitted observation-noise covariance is invalid.")
    return noise


def _white_noise_level(kernel: Any) -> float:
    if isinstance(kernel, WhiteKernel):
        return float(kernel.noise_level)
    if isinstance(kernel, (Sum, Product)):
        return _white_noise_level(kernel.k1) + _white_noise_level(kernel.k2)
    if isinstance(kernel, Exponentiation):
        return _white_noise_level(kernel.kernel)
    return 0.0


def _without_white_noise(kernel: Any) -> Kernel | None:
    if isinstance(kernel, WhiteKernel):
        return None
    if isinstance(kernel, Sum):
        left = _without_white_noise(kernel.k1)
        right = _without_white_noise(kernel.k2)
        if left is None:
            return right
        if right is None:
            return left
        return Sum(left, right)
    if isinstance(kernel, Product):
        left = _without_white_noise(kernel.k1)
        right = _without_white_noise(kernel.k2)
        if left is None or right is None:
            return None
        return Product(left, right)
    if isinstance(kernel, Exponentiation):
        base = _without_white_noise(kernel.kernel)
        return None if base is None else Exponentiation(base, kernel.exponent)
    return kernel


def _kernel_correlations(
    cross: NDArray[np.float64],
    reference_variance: NDArray[np.float64],
    training_covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    training_variance = np.diag(training_covariance)
    denominator = np.sqrt(np.maximum(reference_variance[:, None] * training_variance[None, :], 0.0))
    correlations = np.zeros(len(cross), dtype=float)
    valid = denominator > 0
    values = np.zeros_like(cross)
    values[valid] = np.abs(cross[valid] / denominator[valid])
    correlations[:] = np.max(values, axis=1)
    return np.clip(correlations, 0.0, 1.0)


def _points(values: ArrayLike, name: str) -> NDArray[np.float64]:
    points = np.asarray(values, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] == 0 or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be a finite two-dimensional array.")
    return points


def _require_finite(values: NDArray[np.float64], name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(f"{name} contains non-finite values.")
