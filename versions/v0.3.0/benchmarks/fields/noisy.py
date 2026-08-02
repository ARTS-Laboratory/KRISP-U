"""A smooth trend plus deterministic, spatially correlated roughness."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from benchmarks.fields.smooth import DOMAIN, AuditField


def noisy_field(seed: int = 2024) -> AuditField:
    """Construct a reproducible rough field from band-limited Fourier modes.

    The perturbation is spatially structured rather than independent point noise.
    A factory seed changes the fixed Fourier coefficients while keeping evaluation
    deterministic for every input array.
    """

    rng = np.random.default_rng(seed)
    frequencies = rng.integers(2, 10, size=(28, 2))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=28)
    amplitudes = rng.normal(size=28) / np.sqrt(frequencies[:, 0] + frequencies[:, 1])
    amplitudes /= np.linalg.norm(amplitudes)

    def evaluate(X: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        x, y = values[:, 0], values[:, 1]
        trend = (
            0.45 * np.sin(2.2 * x)
            - 0.35 * np.cos(1.8 * y)
            - 0.2 * x * y
            - 0.35 * np.exp(-3.5 * ((x - 0.3) ** 2 + (y + 0.25) ** 2))
        )
        phase = np.pi * (
            frequencies[:, 0, None] * (x[None, :] + 1.0) / 2.0
            + frequencies[:, 1, None] * (y[None, :] + 1.0) / 2.0
        )
        roughness = 0.28 * np.sum(amplitudes[:, None] * np.sin(phase + phases[:, None]), axis=0)
        return trend + roughness

    return AuditField(
        name="noisy_baseline",
        domain=DOMAIN,
        evaluate=evaluate,
        recommended_plot_limits=(-1.5, 1.3),
        metadata={
            "description": "Smooth trend with a fixed-seed band-limited rough baseline",
            "construction": "28 random Fourier modes with inverse-frequency weighting",
            "seed": int(seed),
            "domain": [[-1.0, 1.0], [-1.0, 1.0]],
        },
    )


__all__ = ["noisy_field"]
