"""Matérn-3/2 and Matérn-5/2 ARD covariances."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from krispu.kernels.base import AnisotropicSpatialKernel


class Matern32ARD(AnisotropicSpatialKernel):
    def _correlation(
        self, distance: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], None]:
        root = np.sqrt(3.0)
        exponential = np.exp(-root * distance)
        return (1.0 + root * distance) * exponential, -3.0 * distance * exponential, None


class Matern52ARD(AnisotropicSpatialKernel):
    def _correlation(
        self, distance: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], None]:
        root = np.sqrt(5.0)
        exponential = np.exp(-root * distance)
        correlation = (1.0 + root * distance + 5.0 * distance**2 / 3.0) * exponential
        derivative = -(5.0 / 3.0) * distance * (1.0 + root * distance) * exponential
        return correlation, derivative, None
