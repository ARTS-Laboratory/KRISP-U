"""Deterministic fields used by the v0.3.0 kernel-selection study."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from benchmarks.fields.smooth import DOMAIN, AuditField


def rough_correlated_field(seed: int = 3101) -> AuditField:
    """Broad smooth structure plus reproducible short-scale correlated modes."""

    frequencies, phases, amplitudes = _fourier_modes(seed, 36, 3, 13)

    def evaluate(X: ArrayLike) -> NDArray[np.float64]:
        values = _points(X)
        x, y = values[:, 0], values[:, 1]
        broad = 0.55 * np.sin(2.0 * x) - 0.35 * np.cos(1.8 * y) - 0.15 * x * y
        rough = _modes(x, y, frequencies, phases, amplitudes, scale=0.32)
        return broad + rough

    return AuditField(
        "rough_correlated",
        DOMAIN,
        evaluate,
        (-1.5, 1.5),
        {
            "description": "Broad smooth component plus short-scale correlated Fourier field",
            "field_family": "rough_single_scale",
            "generating_kernel": "broad_plus_short_correlated",
            "seed": seed,
        },
    )


def rough_multiscale_field(seed: int = 3102) -> AuditField:
    """Field with two distinct correlated spatial scales."""

    long_f, long_p, long_a = _fourier_modes(seed, 20, 1, 5)
    short_f, short_p, short_a = _fourier_modes(seed + 1, 36, 6, 16)

    def evaluate(X: ArrayLike) -> NDArray[np.float64]:
        values = _points(X)
        x, y = values[:, 0], values[:, 1]
        long_component = _modes(x, y, long_f, long_p, long_a, scale=0.52)
        short_component = _modes(x, y, short_f, short_p, short_a, scale=0.20)
        return long_component + short_component

    return AuditField(
        "rough_multiscale",
        DOMAIN,
        evaluate,
        (-1.5, 1.5),
        {
            "description": "Two correlated Fourier components with distinct spatial scales",
            "field_family": "rough_multiscale",
            "generating_kernel": "matern_52_long_plus_matern_12_short",
            "generating_hyperparameters": {
                "long_scale": [0.55, 0.72],
                "short_scale": [0.07, 0.11],
            },
            "seed": seed,
        },
    )


def periodic_field(seed: int = 3103) -> AuditField:
    """Approximately periodic field with a localized non-periodic deviation."""

    rng = np.random.default_rng(seed)
    phase = float(rng.uniform(-np.pi, np.pi))

    def evaluate(X: ArrayLike) -> NDArray[np.float64]:
        values = _points(X)
        x, y = values[:, 0], values[:, 1]
        return (
            0.65 * np.sin(2.0 * np.pi * (x + 1.0) / 1.6 + phase)
            + 0.35 * np.cos(2.0 * np.pi * (y + 1.0) / 1.25)
            + 0.20 * np.sin(2.0 * np.pi * (x + y) / 1.9)
        )

    return AuditField(
        "periodic",
        DOMAIN,
        evaluate,
        (-1.4, 1.4),
        {
            "description": "Reproducible two-dimensional periodic field",
            "field_family": "periodic",
            "generating_kernel": "periodic_times_matern_32",
            "seed": seed,
        },
    )


def trend_plus_local_field(seed: int = 3104) -> AuditField:
    """Global linear trend plus a correlated local residual feature."""

    del seed

    def evaluate(X: ArrayLike) -> NDArray[np.float64]:
        values = _points(X)
        x, y = values[:, 0], values[:, 1]
        trend = 0.55 * x - 0.35 * y + 0.12 * x * y
        local = 0.55 * np.exp(-((x + 0.35) ** 2 / 0.08) - ((y - 0.25) ** 2 / 0.035))
        return trend + local + 0.15 * np.sin(2.5 * x - 0.5 * y)

    return AuditField(
        "trend_plus_local",
        DOMAIN,
        evaluate,
        (-1.1, 1.1),
        {
            "description": "Linear trend with a localized correlated residual",
            "field_family": "trend_plus_local",
            "generating_kernel": "linear_plus_matern_32",
        },
    )


def _fourier_modes(seed: int, count: int, low: int, high: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    frequencies = rng.integers(low, high, size=(count, 2))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=count)
    amplitudes = rng.normal(size=count) / np.sqrt(np.sum(frequencies, axis=1))
    amplitudes /= np.linalg.norm(amplitudes)
    return frequencies, phases, amplitudes


def _modes(
    x: np.ndarray,
    y: np.ndarray,
    frequencies: np.ndarray,
    phases: np.ndarray,
    amplitudes: np.ndarray,
    scale: float,
) -> np.ndarray:
    phase = np.pi * (
        frequencies[:, 0, None] * (x[None, :] + 1.0) / 2.0
        + frequencies[:, 1, None] * (y[None, :] + 1.0) / 2.0
    )
    return scale * np.sum(amplitudes[:, None] * np.sin(phase + phases[:, None]), axis=0)


def _points(X: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(X, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("kernel-study fields require two-dimensional coordinates.")
    return values
