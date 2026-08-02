"""Shared uncertainty data structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class UncertaintyComponents:
    """The separate terms that form the canonical KRISP-U field."""

    loo_mean: NDArray[np.float64]
    jackknife_std: NDArray[np.float64]
    loo_calibration_factor: float
    calibrated_posterior_std: NDArray[np.float64]
    combined_std: NDArray[np.float64]
