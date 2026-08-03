"""Scikit-learn-compatible base class for global anisotropic spatial kernels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.gaussian_process.kernels import Hyperparameter, Kernel

from krispu.kernels.distance import scaled_distance_parts, validate_points


class AnisotropicSpatialKernel(Kernel, ABC):
    """One global spatial covariance family with one ARD scale vector."""

    def __init__(
        self,
        amplitude: float = 1.0,
        amplitude_bounds: tuple[float, float] | str = (1e-3, 5.0),
        length_scale: ArrayLike = (0.25,),
        length_scale_bounds: ArrayLike | str = (0.02, 2.0),
    ) -> None:
        self.amplitude = float(amplitude)
        self.amplitude_bounds = amplitude_bounds
        self.length_scale = np.asarray(length_scale, dtype=float)
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_amplitude(self) -> Hyperparameter:
        return Hyperparameter("amplitude", "numeric", self.amplitude_bounds)

    @property
    def hyperparameter_length_scale(self) -> Hyperparameter:
        bounds = np.asarray(self.length_scale_bounds, dtype=float)
        if bounds.ndim == 1:
            bounds = np.broadcast_to(bounds, (len(self.length_scale), 2)).copy()
        return Hyperparameter("length_scale", "numeric", bounds, len(self.length_scale))

    @property
    def theta(self) -> NDArray[np.float64]:
        return np.log(np.concatenate(([self.amplitude], self.length_scale)))

    @theta.setter
    def theta(self, theta: ArrayLike) -> None:
        values = np.exp(np.asarray(theta, dtype=float))
        expected = 1 + len(self.length_scale)
        if values.shape != (expected,):
            raise ValueError("theta has an invalid size for this ARD kernel.")
        self.amplitude = float(values[0])
        self.length_scale = values[1:].copy()

    @property
    def bounds(self) -> NDArray[np.float64]:
        amplitude = np.asarray(self.hyperparameter_amplitude.bounds, dtype=float)
        scales = np.asarray(self.hyperparameter_length_scale.bounds, dtype=float)
        return np.log(np.vstack((amplitude, scales)))

    def is_stationary(self) -> bool:
        return True

    def diag(self, X: ArrayLike) -> NDArray[np.float64]:
        points = validate_points(X, "X")
        return np.full(len(points), self.amplitude, dtype=float)

    def __call__(
        self,
        X: ArrayLike,
        Y: ArrayLike | None = None,
        eval_gradient: bool = False,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        left = validate_points(X, "X")
        right = None if Y is None else validate_points(Y, "Y")
        distance, contributions = scaled_distance_parts(left, right, self.length_scale)
        covariance, derivative_distance, extra_gradient = self._correlation(distance)
        covariance = self.amplitude * covariance
        if not eval_gradient:
            return covariance
        if Y is not None:
            raise ValueError("Gradient can only be evaluated when Y is None.")
        safe_distance = np.where(distance > 0.0, distance, 1.0)
        distance_gradient = -contributions / safe_distance[:, :, None]
        distance_gradient[distance == 0.0, :] = 0.0
        gradients = self.amplitude * derivative_distance[:, :, None] * distance_gradient
        if extra_gradient is not None:
            gradients = np.concatenate((covariance[:, :, None], gradients, extra_gradient), axis=2)
        else:
            gradients = np.concatenate((covariance[:, :, None], gradients), axis=2)
        return covariance, gradients

    @abstractmethod
    def _correlation(
        self, distance: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64] | None,
    ]:
        """Return correlation, dcorrelation/dr, and extra parameter gradients."""

    def _get_param_names(self) -> list[str]:
        return ["amplitude", "amplitude_bounds", "length_scale", "length_scale_bounds"]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(amplitude={self.amplitude:.6g}, "
            f"length_scale={self.length_scale!r})"
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "amplitude": self.amplitude,
            "amplitude_bounds": self.amplitude_bounds,
            "length_scale": self.length_scale,
            "length_scale_bounds": self.length_scale_bounds,
        }

    def set_params(self, **params: Any) -> AnisotropicSpatialKernel:
        for name, value in params.items():
            if name not in self.get_params():
                raise ValueError(f"Invalid parameter {name!r} for {type(self).__name__}.")
            setattr(self, name, value)
        self.length_scale = np.asarray(self.length_scale, dtype=float)
        return self
