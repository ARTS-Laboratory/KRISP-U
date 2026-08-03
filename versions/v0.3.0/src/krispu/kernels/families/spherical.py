"""Spherical ARD covariance, valid in dimensions one through three."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from krispu.kernels.base import AnisotropicSpatialKernel


class SphericalARD(AnisotropicSpatialKernel):
    """Spherical covariance with the standard positive-definite d <= 3 limit."""

    def _check_dimension(self, distance: NDArray[np.float64]) -> None:
        if len(self.length_scale) > 3:
            raise ValueError("SphericalARD is positive definite only through dimension three.")

    def _correlation(
        self, distance: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], None]:
        self._check_dimension(distance)
        inside = distance < 1.0
        correlation = np.where(inside, 1.0 - 1.5 * distance + 0.5 * distance**3, 0.0)
        derivative = np.where(inside, -1.5 + 1.5 * distance**2, 0.0)
        return correlation, derivative, None
