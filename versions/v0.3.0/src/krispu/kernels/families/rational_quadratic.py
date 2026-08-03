"""ARD rational-quadratic covariance."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.gaussian_process.kernels import Hyperparameter

from krispu.kernels.base import AnisotropicSpatialKernel


class RationalQuadraticARD(AnisotropicSpatialKernel):
    def __init__(self, alpha: float = 1.0, alpha_bounds: tuple[float, float] | str = (1e-3, 100.0), **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.alpha_bounds = alpha_bounds

    @property
    def hyperparameter_alpha(self) -> Hyperparameter:
        return Hyperparameter("alpha", "numeric", self.alpha_bounds, 1)

    @property
    def theta(self) -> NDArray[np.float64]:
        return np.log(np.concatenate(([self.amplitude], self.length_scale, [self.alpha])))

    @theta.setter
    def theta(self, theta: NDArray[np.float64]) -> None:
        values = np.exp(np.asarray(theta, dtype=float))
        expected = len(self.length_scale) + 2
        if values.shape != (expected,):
            raise ValueError("theta has an invalid size for RationalQuadraticARD.")
        self.amplitude = float(values[0])
        self.length_scale = values[1:-1].copy()
        self.alpha = float(values[-1])

    @property
    def bounds(self) -> NDArray[np.float64]:
        return np.log(np.vstack((self.hyperparameter_amplitude.bounds, self.hyperparameter_length_scale.bounds, self.hyperparameter_alpha.bounds)))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        params = super().get_params(deep)
        params.update({"alpha": self.alpha, "alpha_bounds": self.alpha_bounds})
        return params

    def set_params(self, **params: object) -> RationalQuadraticARD:
        super().set_params(**{key: value for key, value in params.items() if key not in {"alpha", "alpha_bounds"}})
        if "alpha" in params:
            self.alpha = float(params["alpha"])
        if "alpha_bounds" in params:
            self.alpha_bounds = params["alpha_bounds"]  # type: ignore[assignment]
        return self

    def _correlation(
        self, distance: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")
        z = 1.0 + distance**2 / (2.0 * self.alpha)
        correlation = z ** (-self.alpha)
        derivative_distance = -distance * correlation / z
        alpha_gradient = correlation * (-self.alpha * np.log(z) + (z - 1.0) / z)
        return correlation, derivative_distance, alpha_gradient[:, :, None]
