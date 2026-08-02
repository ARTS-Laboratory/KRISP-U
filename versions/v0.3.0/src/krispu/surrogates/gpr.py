"""Stable Matérn-3/2 ARD Gaussian-process surrogate.

The public ``GPRSurrogate`` works in normalized coordinates and reports
predictions in the original response units.  A fitted kernel is deliberately
reused without re-optimization by the LOO backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from krispu.config import GPRConfig


@dataclass(frozen=True)
class ResponseStandardizer:
    """Affine response transform used by both full and LOO models."""

    mean: float
    scale: float

    @classmethod
    def fit(cls, y: ArrayLike, epsilon: float = 1e-12) -> ResponseStandardizer:
        values = np.asarray(y, dtype=float).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("y must contain at least one finite response.")
        scale = float(np.std(values))
        if scale < epsilon:
            scale = 1.0
        return cls(float(np.mean(values)), scale)

    def transform(self, y: ArrayLike) -> NDArray[np.float64]:
        return (np.asarray(y, dtype=float) - self.mean) / self.scale

    def inverse_transform(self, y: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(y, dtype=float) * self.scale + self.mean

    def inverse_std(self, std: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(std, dtype=float) * self.scale


class GPRSurrogate:
    """A response-standardized, normalized-coordinate GPR."""

    def __init__(self, config: GPRConfig | None = None) -> None:
        self.config = config or GPRConfig()
        self.model_: GaussianProcessRegressor | None = None
        self.standardizer_: ResponseStandardizer | None = None
        self.X_train_: NDArray[np.float64] | None = None
        self.y_train_: NDArray[np.float64] | None = None
        self.observation_variances_: NDArray[np.float64] | None = None
        self.kernel_: Any | None = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        observation_variances: ArrayLike | None = None,
    ) -> GPRSurrogate:
        points = _validate_X(X)
        values = np.asarray(y, dtype=float).reshape(-1)
        if len(points) != len(values) or len(points) < 1:
            raise ValueError("X and y must have the same non-zero length.")
        if not np.all(np.isfinite(values)):
            raise ValueError("y must contain only finite values.")
        standardizer = ResponseStandardizer.fit(values, self.config.response_epsilon)
        variances = self._validate_noise(observation_variances, len(values), standardizer.scale)
        model = self._make_model(
            self._initial_kernel(points.shape[1]),
            optimize=self.config.optimize_hyperparameters,
            alpha=self._alpha(variances),
        )
        model.fit(points, standardizer.transform(values))
        self.model_ = model
        self.standardizer_ = standardizer
        self.X_train_ = points.copy()
        self.y_train_ = values.copy()
        self.observation_variances_ = None if variances is None else variances.copy()
        self.kernel_ = clone(model.kernel_)
        return self

    @property
    def log_marginal_likelihood(self) -> float:
        """Return the fitted log marginal likelihood."""

        if self.model_ is None:
            raise ValueError("Call fit() before accessing the log marginal likelihood.")
        value = float(self.model_.log_marginal_likelihood_value_)
        if not np.isfinite(value):
            raise FloatingPointError("The fitted log marginal likelihood is non-finite.")
        return value

    def fit_fixed_kernel(
        self,
        X: ArrayLike,
        y: ArrayLike,
        observation_variances: ArrayLike | None = None,
        standardizer: ResponseStandardizer | None = None,
        frozen_kernel: Any | None = None,
    ) -> GPRSurrogate:
        """Fit a fold with fixed full-model hyperparameters.

        ``standardizer`` and ``frozen_kernel`` must come from the complete
        observation fit when this method is used for LOO.
        """

        points = _validate_X(X)
        values = np.asarray(y, dtype=float).reshape(-1)
        if len(points) != len(values) or len(points) < 1:
            raise ValueError("X and y must have the same non-zero length.")
        if not np.all(np.isfinite(values)):
            raise ValueError("y must contain only finite values.")
        if standardizer is None:
            standardizer = ResponseStandardizer.fit(values, self.config.response_epsilon)
        variances = self._validate_noise(observation_variances, len(values), standardizer.scale)
        kernel = clone(
            frozen_kernel if frozen_kernel is not None else self._initial_kernel(points.shape[1])
        )
        model = self._make_model(kernel, optimize=False, alpha=self._alpha(variances))
        model.fit(points, standardizer.transform(values))
        self.model_ = model
        self.standardizer_ = standardizer
        self.X_train_ = points.copy()
        self.y_train_ = values.copy()
        self.observation_variances_ = None if variances is None else variances.copy()
        self.kernel_ = clone(model.kernel_)
        return self

    def predict(self, X: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self.model_ is None or self.standardizer_ is None:
            raise ValueError("Call fit() before predict().")
        points = _validate_X(X)
        mean, std = self.model_.predict(points, return_std=True)
        mean_original = self.standardizer_.inverse_transform(mean).reshape(-1)
        std_original = self.standardizer_.inverse_std(std).reshape(-1)
        std_original = np.maximum(std_original, 0.0)
        if not np.all(np.isfinite(mean_original)) or not np.all(np.isfinite(std_original)):
            raise FloatingPointError("GPR produced a non-finite prediction.")
        return mean_original, std_original

    @property
    def standardizer(self) -> ResponseStandardizer:
        if self.standardizer_ is None:
            raise ValueError("Call fit() before accessing the response standardizer.")
        return self.standardizer_

    @property
    def frozen_kernel(self) -> Any:
        if self.kernel_ is None:
            raise ValueError("Call fit() before accessing the fitted kernel.")
        return clone(self.kernel_)

    def _initial_kernel(self, dimension: int) -> Any:
        if self.config.kernel is not None:
            return clone(self.config.kernel)
        length_scale = np.full(dimension, self.config.length_scale_initial, dtype=float)
        length_bounds = self.config.length_scale_bounds
        base = Matern(length_scale=length_scale, length_scale_bounds=length_bounds, nu=1.5)
        kernel: Any = (
            ConstantKernel(
                self.config.constant_value_initial,
                constant_value_bounds=self.config.constant_value_bounds,
            )
            * base
        )
        if self.config.noise_mode == "noisy" and self.config.fit_white_noise:
            kernel = kernel + WhiteKernel(
                noise_level=self.config.white_noise_initial,
                noise_level_bounds=self.config.white_noise_bounds,
            )
        return kernel

    def _make_model(
        self, kernel: Any, optimize: bool, alpha: float | NDArray[np.float64]
    ) -> GaussianProcessRegressor:
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=False,
            optimizer="fmin_l_bfgs_b" if optimize else None,
            n_restarts_optimizer=self.config.n_restarts_optimizer if optimize else 0,
            random_state=self.config.random_state,
        )

    def _validate_noise(
        self,
        variances: ArrayLike | None,
        n: int,
        response_scale: float,
    ) -> NDArray[np.float64] | None:
        if self.config.noise_mode == "deterministic":
            if variances is not None:
                raise ValueError("observation variances require noise_mode='noisy'.")
            return None
        if variances is None:
            scalar = self.config.observation_noise_variance
            if scalar is None:
                return None
            variances = np.full(n, scalar, dtype=float)
        result = np.asarray(variances, dtype=float).reshape(-1)
        if len(result) != n or not np.all(np.isfinite(result)) or np.any(result < 0):
            raise ValueError("observation variances must be finite, non-negative, and match y.")
        return result / response_scale**2

    def _alpha(self, variances: NDArray[np.float64] | None) -> float | NDArray[np.float64]:
        if variances is None:
            return self.config.alpha
        return self.config.alpha + variances


def _validate_X(X: ArrayLike) -> NDArray[np.float64]:
    points = np.asarray(X, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] == 0 or not np.all(np.isfinite(points)):
        raise ValueError("X must be a finite two-dimensional array.")
    return points
