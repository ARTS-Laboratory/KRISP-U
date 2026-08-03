"""Compactly supported Wendland C2 ARD covariance."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from krispu.kernels.base import AnisotropicSpatialKernel


class WendlandC2ARD(AnisotropicSpatialKernel):
    """Wendland C2 covariance, positive definite in dimensions one through three."""

    def _correlation(
        self, distance: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], None]:
        if len(self.length_scale) > 3:
            raise ValueError("WendlandC2ARD is positive definite only through dimension three.")
        inside = distance < 1.0
        remainder = np.maximum(1.0 - distance, 0.0)
        correlation = np.where(inside, remainder**4 * (4.0 * distance + 1.0), 0.0)
        derivative = np.where(inside, -20.0 * distance * remainder**3, 0.0)
        return correlation, derivative, None
