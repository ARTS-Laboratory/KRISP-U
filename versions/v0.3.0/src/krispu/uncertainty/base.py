"""Shared uncertainty data structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class UncertaintyComponents:
    """The separate terms that form the canonical KRISP-U field."""

    loo_mean: NDArray[np.float64]
    loo_field_sensitivity: NDArray[np.float64]
    kernel_support_deficit: NDArray[np.float64]
    krispu_uncertainty: NDArray[np.float64]
    posterior_std: NDArray[np.float64]
