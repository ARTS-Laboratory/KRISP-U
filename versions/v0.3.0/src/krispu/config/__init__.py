"""Small, explicit configuration objects for the v0.3.0 scientific core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class BufferedJackknifeConfig:
    mode: str = "median_nearest_neighbor"
    multiplier: float = 1.0
    minimum_radius: float = 0.025
    maximum_radius: float = 0.20
    minimum_training_points: int = 3

    def __post_init__(self) -> None:
        if self.mode != "median_nearest_neighbor":
            raise ValueError("jackknife.buffer.mode must be median_nearest_neighbor.")
        if self.multiplier < 0 or not np.isfinite(self.multiplier):
            raise ValueError("jackknife.buffer.multiplier must be finite and non-negative.")
        if not (0 < self.minimum_radius <= self.maximum_radius):
            raise ValueError("jackknife buffer radius bounds must be positive and increasing.")
        if self.minimum_training_points < 1:
            raise ValueError("jackknife.buffer.minimum_training_points must be positive.")


@dataclass(frozen=True)
class GPRConfig:
    """Configuration for normalized-coordinate GP fitting."""

    noise_mode: str = "deterministic"
    alpha: float = 1e-10
    observation_noise_variance: float | None = None
    fit_white_noise: bool = False
    white_noise_initial: float = 1e-6
    white_noise_bounds: tuple[float, float] = (1e-10, 1.0)
    length_scale_initial: float | tuple[float, ...] | NDArray[np.float64] = 0.25
    length_scale_bounds: tuple[float, float] | tuple[tuple[float, float], ...] = (0.02, 2.0)
    constant_value_initial: float = 1.0
    constant_value_bounds: tuple[float, float] = (1e-3, 1e3)
    n_restarts_optimizer: int = 0
    random_state: int | None = 0
    response_epsilon: float = 1e-12
    kernel: Any | None = None
    optimize_hyperparameters: bool = True
    jackknife: BufferedJackknifeConfig = BufferedJackknifeConfig()

    def __post_init__(self) -> None:
        if self.noise_mode not in {"deterministic", "noisy"}:
            raise ValueError("noise_mode must be 'deterministic' or 'noisy'.")
        if self.alpha <= 0 or not np.isfinite(self.alpha):
            raise ValueError("alpha must be a finite positive scalar.")
        if self.observation_noise_variance is not None and self.observation_noise_variance < 0:
            raise ValueError("observation_noise_variance must be non-negative.")
        if self.noise_mode == "deterministic" and self.fit_white_noise:
            raise ValueError("fit_white_noise is only valid in noisy mode.")
        if self.white_noise_initial <= 0 or self.n_restarts_optimizer < 0:
            raise ValueError("noise initial value and optimizer restarts must be non-negative/positive.")
        _validate_pair(self.white_noise_bounds, "white_noise_bounds")
        _validate_pair(self.constant_value_bounds, "constant_value_bounds")
        if self.constant_value_initial <= 0 or self.response_epsilon <= 0:
            raise ValueError("constant_value_initial and response_epsilon must be positive.")

    def resolved_length_scales(self, dimension: int) -> NDArray[np.float64]:
        values = np.asarray(self.length_scale_initial, dtype=float)
        if values.ndim == 0:
            values = np.full(dimension, float(values), dtype=float)
        if values.shape != (dimension,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("length_scale_initial must resolve to one positive value per dimension.")
        return values.copy()

    def resolved_length_scale_bounds(self, dimension: int) -> NDArray[np.float64]:
        values = np.asarray(self.length_scale_bounds, dtype=float)
        if values.shape == (2,):
            values = np.broadcast_to(values, (dimension, 2)).copy()
        if values.shape != (dimension, 2) or not np.all(np.isfinite(values)):
            raise ValueError("length_scale_bounds must resolve to one pair per dimension.")
        if np.any(values[:, 0] <= 0) or np.any(values[:, 0] >= values[:, 1]):
            raise ValueError("length_scale_bounds must contain increasing positive pairs.")
        return values


def _validate_pair(value: ArrayLike, name: str) -> None:
    values = np.asarray(value, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)) or not (0 < values[0] < values[1]):
        raise ValueError(f"{name} must be an increasing positive pair.")


GprConfig = GPRConfig

__all__ = ["BufferedJackknifeConfig", "GPRConfig", "GprConfig"]
