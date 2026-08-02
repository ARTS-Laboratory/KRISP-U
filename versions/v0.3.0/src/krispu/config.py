"""Small, explicit configuration objects for the v0.3.0 scientific core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GPRConfig:
    """Configuration for the fixed-hyperparameter LOO GPR workflow.

    Coordinates are normalized before this configuration is used.  Therefore
    the length-scale bounds are in normalized coordinate units.
    """

    noise_mode: str = "deterministic"
    alpha: float = 1e-10
    observation_noise_variance: float | None = None
    fit_white_noise: bool = False
    white_noise_initial: float = 1e-6
    white_noise_bounds: tuple[float, float] = (1e-10, 1.0)
    length_scale_initial: float = 0.25
    length_scale_bounds: tuple[float, float] = (0.02, 2.0)
    constant_value_initial: float = 1.0
    constant_value_bounds: tuple[float, float] = (1e-3, 1e3)
    n_restarts_optimizer: int = 0
    random_state: int | None = 0
    response_epsilon: float = 1e-12
    # When provided, this is the complete kernel template to use instead of
    # the historical Matérn-(3/2) default.  It is cloned before fitting.
    kernel: Any | None = None
    optimize_hyperparameters: bool = True

    def __post_init__(self) -> None:
        if self.noise_mode not in {"deterministic", "noisy"}:
            raise ValueError("noise_mode must be 'deterministic' or 'noisy'.")
        if self.alpha <= 0 or not np.isfinite(self.alpha):
            raise ValueError("alpha must be a finite positive scalar.")
        if self.observation_noise_variance is not None and self.observation_noise_variance < 0:
            raise ValueError("observation_noise_variance must be non-negative.")
        if self.noise_mode == "deterministic" and self.fit_white_noise:
            raise ValueError("fit_white_noise is only valid in noisy mode.")
        if self.white_noise_initial <= 0:
            raise ValueError("white_noise_initial must be positive.")
        if len(self.white_noise_bounds) != 2 or not (
            0 < self.white_noise_bounds[0] < self.white_noise_bounds[1]
        ):
            raise ValueError("white_noise_bounds must be (positive_lower, upper).")
        if self.length_scale_initial <= 0:
            raise ValueError("length_scale_initial must be positive.")
        if len(self.length_scale_bounds) != 2 or not (
            0 < self.length_scale_bounds[0] < self.length_scale_bounds[1]
        ):
            raise ValueError("length_scale_bounds must be (positive_lower, upper).")
        if self.n_restarts_optimizer < 0:
            raise ValueError("n_restarts_optimizer must be non-negative.")
        if self.response_epsilon <= 0:
            raise ValueError("response_epsilon must be positive.")


# The spelling used in some scientific code is retained as a descriptive name,
# not as a historical compatibility alias.
GprConfig = GPRConfig
