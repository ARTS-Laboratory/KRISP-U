"""Exponential ARD covariance."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from krispu.kernels.base import AnisotropicSpatialKernel


class ExponentialARD(AnisotropicSpatialKernel):
    def _correlation(
        self, distance: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], None]:
        correlation = np.exp(-distance)
        return correlation, -correlation, None
