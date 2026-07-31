"""Gaussian-process model helpers for KRISP-U."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneOut

DEFAULT_KERNEL_FAMILIES = (
    "matern_0.5",
    "matern_1.5",
    "matern_2.5",
    "rbf",
    "rational_quadratic",
)


@dataclass(frozen=True)
class VariogramSummary:
    """Empirical variogram-derived prior summary."""

    dimension: int
    n_samples: int
    n_pairs: int
    distance_min: float
    distance_median: float
    distance_max: float
    semivariance_min: float
    semivariance_median: float
    semivariance_max: float
    nugget: float
    sill: float
    range: float
    length_scale: tuple[float, ...]
    length_scale_bounds: tuple[tuple[float, float], ...]
    noise_level: float
    fallback_used: bool = False

    def to_dict(
        self,
    ) -> dict[str, float | int | bool | list[float] | list[list[float]]]:
        """Return a JSON-friendly representation."""

        return {
            "dimension": self.dimension,
            "n_samples": self.n_samples,
            "n_pairs": self.n_pairs,
            "distance_min": self.distance_min,
            "distance_median": self.distance_median,
            "distance_max": self.distance_max,
            "semivariance_min": self.semivariance_min,
            "semivariance_median": self.semivariance_median,
            "semivariance_max": self.semivariance_max,
            "nugget": self.nugget,
            "sill": self.sill,
            "range": self.range,
            "length_scale": list(self.length_scale),
            "length_scale_bounds": [list(pair) for pair in self.length_scale_bounds],
            "noise_level": self.noise_level,
            "fallback_used": self.fallback_used,
        }


@dataclass(frozen=True)
class KernelCandidateScore:
    """One candidate kernel's model-selection metrics."""

    family: str
    kernel_repr: str
    cv_rmse: float
    cv_r2: float
    log_marginal_likelihood: float
    coverage_95: float
    coverage_penalty: float
    score: float
    error: str | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        """Return a JSON-friendly representation."""

        return {
            "family": self.family,
            "kernel_repr": self.kernel_repr,
            "cv_rmse": self.cv_rmse,
            "cv_r2": self.cv_r2,
            "log_marginal_likelihood": self.log_marginal_likelihood,
            "coverage_95": self.coverage_95,
            "coverage_penalty": self.coverage_penalty,
            "score": self.score,
            "error": self.error,
        }


@dataclass(frozen=True)
class KernelPriorResult:
    """Selected kernel and model-selection evidence for one fit."""

    selected_family: str
    selected_kernel: Any
    selected_kernel_repr: str
    variogram: VariogramSummary
    candidate_scores: list[KernelCandidateScore]

    @property
    def best_score(self) -> float:
        """Return the lowest finite model-selection score."""

        scores = self._finite_scores()
        if not scores:
            return float("nan")
        return scores[0]

    @property
    def second_best_score(self) -> float:
        """Return the second-lowest finite model-selection score."""

        scores = self._finite_scores()
        if len(scores) < 2:
            return float("nan")
        return scores[1]

    @property
    def score_margin(self) -> float:
        """Return the score gap between the best and runner-up kernels."""

        if not np.isfinite(self.best_score) or not np.isfinite(self.second_best_score):
            return float("nan")
        return self.second_best_score - self.best_score

    def _finite_scores(self) -> list[float]:
        return sorted(
            float(score.score)
            for score in self.candidate_scores
            if score.error is None and np.isfinite(score.score)
        )

    def to_dict(
        self,
    ) -> dict[
        str,
        str
        | float
        | dict[str, float | int | bool | list[float] | list[list[float]]]
        | list[dict[str, float | str | None]],
    ]:
        """Return a JSON-friendly representation."""

        return {
            "selected_family": self.selected_family,
            "selected_kernel_repr": self.selected_kernel_repr,
            "best_score": self.best_score,
            "second_best_score": self.second_best_score,
            "score_margin": self.score_margin,
            "variogram": self.variogram.to_dict(),
            "candidate_scores": [score.to_dict() for score in self.candidate_scores],
        }


@dataclass(frozen=True)
class KernelPriorConfig:
    """Configuration for empirical kernel-prior optimization."""

    enabled: bool = True
    candidate_families: tuple[str, ...] = DEFAULT_KERNEL_FAMILIES
    coverage_target: float = 0.95
    cv_rmse_weight: float = 1.0
    lml_weight: float = 0.05
    coverage_weight: float = 0.25
    max_cv_splits: int = 5
    random_state: int | None = None
    max_variogram_pairs: int = 5000


@dataclass(frozen=True)
class GprConfig:
    """Configuration for the default Gaussian-process surrogate."""

    alpha: float = 1e-8
    noise_level: float = 1e-6
    normalize_y: bool = True
    n_restarts_optimizer: int = 1
    random_state: int | None = None
    kernel: Any | None = None
    adaptive_kernel: bool = True
    kernel_prior_config: KernelPriorConfig | None = None


def make_default_gpr(config: GprConfig | None = None) -> GaussianProcessRegressor:
    """Create the fixed fallback KRISP-U Gaussian-process regressor."""

    config = config or GprConfig()
    kernel = config.kernel or make_fixed_default_kernel(config)
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=config.alpha,
        normalize_y=config.normalize_y,
        n_restarts_optimizer=config.n_restarts_optimizer,
        random_state=config.random_state,
    )


def make_fixed_default_kernel(config: GprConfig | None = None) -> Any:
    """Return the historical fixed KRISP-U kernel."""

    config = config or GprConfig(adaptive_kernel=False)
    return ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * Matern(
        length_scale=1.0,
        length_scale_bounds=(1e-3, 1e3),
        nu=2.5,
    ) + WhiteKernel(
        noise_level=config.noise_level,
        noise_level_bounds=(1e-10, 1e1),
    )


def clone_gpr(model: GaussianProcessRegressor) -> GaussianProcessRegressor:
    """Return a fresh clone of a Gaussian-process model."""

    return clone(model)


def estimate_variogram_summary(
    X: ArrayLike,
    y: ArrayLike,
    bounds: ArrayLike | None = None,
    random_state: int | np.random.Generator | None = None,
    max_pairs: int = 5000,
) -> VariogramSummary:
    """Estimate empirical variogram quantities used as kernel priors."""

    points = np.asarray(X, dtype=float)
    values = np.asarray(y, dtype=float).reshape(-1)
    if points.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if len(points) != len(values):
        raise ValueError("X and y must contain the same number of rows.")
    if len(points) < 2:
        raise ValueError("At least two observations are required.")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
        raise ValueError("X and y must contain only finite values.")

    span = _domain_span(points, bounds)
    length_scale_bounds = tuple(
        (float(max(width * 1e-3, 1e-6)), float(max(width * 10.0, 1e-3)))
        for width in span
    )
    y_variance = float(np.var(values))
    sill = max(y_variance, 1e-12)
    fallback = len(points) < 4

    pair_i, pair_j = np.triu_indices(len(points), k=1)
    n_pairs_total = len(pair_i)
    if n_pairs_total == 0:
        fallback = True
    if n_pairs_total > max_pairs:
        rng = _rng(random_state)
        chosen = rng.choice(n_pairs_total, size=max_pairs, replace=False)
        pair_i = pair_i[chosen]
        pair_j = pair_j[chosen]

    distances = np.linalg.norm(points[pair_i] - points[pair_j], axis=1)
    semivariances = 0.5 * (values[pair_i] - values[pair_j]) ** 2
    finite = np.isfinite(distances) & np.isfinite(semivariances) & (distances > 0.0)
    distances = distances[finite]
    semivariances = semivariances[finite]
    if distances.size < 3:
        fallback = True

    if fallback:
        distance_min = 0.0
        distance_median = float(np.linalg.norm(span) * 0.5)
        distance_max = float(np.linalg.norm(span))
        semivariance_min = 0.0
        semivariance_median = sill
        semivariance_max = sill
        nugget = max(0.01 * sill, 1e-12)
        variogram_range = max(distance_median, 1e-6)
    else:
        distance_min = float(np.min(distances))
        distance_median = float(np.median(distances))
        distance_max = float(np.max(distances))
        semivariance_min = float(np.min(semivariances))
        semivariance_median = float(np.median(semivariances))
        semivariance_max = float(np.max(semivariances))
        nugget = float(max(np.quantile(semivariances, 0.05), 1e-12))
        sill = float(max(sill, semivariance_median, nugget, 1e-12))
        high_semivariance = semivariances >= 0.95 * sill
        variogram_range = (
            float(np.min(distances[high_semivariance]))
            if np.any(high_semivariance)
            else distance_median
        )
        variogram_range = max(variogram_range, distance_min, 1e-6)

    length_scale = tuple(
        float(np.clip(0.5 * width, low, high))
        for width, (low, high) in zip(span, length_scale_bounds, strict=True)
    )
    noise_level = float(max(min(nugget, sill), 1e-10))
    return VariogramSummary(
        dimension=points.shape[1],
        n_samples=len(points),
        n_pairs=int(min(n_pairs_total, max_pairs)),
        distance_min=distance_min,
        distance_median=distance_median,
        distance_max=distance_max,
        semivariance_min=semivariance_min,
        semivariance_median=semivariance_median,
        semivariance_max=semivariance_max,
        nugget=float(nugget),
        sill=float(sill),
        range=float(variogram_range),
        length_scale=length_scale,
        length_scale_bounds=length_scale_bounds,
        noise_level=noise_level,
        fallback_used=fallback,
    )


def make_kernel_candidates(
    variogram: VariogramSummary,
    config: KernelPriorConfig | None = None,
) -> list[tuple[str, Any]]:
    """Return empirical-prior kernel candidates."""

    config = config or KernelPriorConfig()
    candidates: list[tuple[str, Any]] = []
    for family in config.candidate_families:
        normalized = family.lower().replace("-", "_")
        candidates.append((normalized, _kernel_for_family(normalized, variogram)))
    return candidates


def fit_prior_optimized_gpr(
    X: ArrayLike,
    y: ArrayLike,
    gpr_config: GprConfig | None = None,
    kernel_prior_config: KernelPriorConfig | None = None,
    bounds: ArrayLike | None = None,
) -> tuple[GaussianProcessRegressor, KernelPriorResult]:
    """Fit a GPR after empirical-prior kernel selection."""

    gpr_config = gpr_config or GprConfig()
    kernel_prior_config = (
        kernel_prior_config
        or gpr_config.kernel_prior_config
        or KernelPriorConfig(random_state=gpr_config.random_state)
    )
    points = np.asarray(X, dtype=float)
    values = np.asarray(y, dtype=float).reshape(-1)
    variogram = estimate_variogram_summary(
        points,
        values,
        bounds=bounds,
        random_state=kernel_prior_config.random_state or gpr_config.random_state,
        max_pairs=kernel_prior_config.max_variogram_pairs,
    )
    candidate_scores: list[KernelCandidateScore] = []
    fitted_models: dict[str, GaussianProcessRegressor] = {}
    response_range = float(np.max(values) - np.min(values))
    if response_range <= 1e-12:
        response_range = 1.0

    for family, kernel in make_kernel_candidates(variogram, kernel_prior_config):
        score, fitted = _score_kernel_candidate(
            family,
            kernel,
            points,
            values,
            response_range,
            gpr_config,
            kernel_prior_config,
        )
        candidate_scores.append(score)
        if fitted is not None:
            fitted_models[family] = fitted

    finite_scores = [score for score in candidate_scores if np.isfinite(score.score)]
    if finite_scores:
        selected_score = min(finite_scores, key=lambda item: item.score)
        selected_model = fitted_models[selected_score.family]
    else:
        fallback_family = "matern_2.5"
        fallback_kernel = _kernel_for_family(fallback_family, variogram)
        selected_model = _make_gpr(fallback_kernel, gpr_config)
        selected_model.fit(points, values)
        selected_score = KernelCandidateScore(
            family=fallback_family,
            kernel_repr=str(selected_model.kernel_),
            cv_rmse=float("nan"),
            cv_r2=float("nan"),
            log_marginal_likelihood=float(
                getattr(selected_model, "log_marginal_likelihood_value_", float("nan"))
            ),
            coverage_95=float("nan"),
            coverage_penalty=float("nan"),
            score=float("inf"),
            error="all candidate kernels failed; used fallback",
        )
        candidate_scores.append(selected_score)

    result = KernelPriorResult(
        selected_family=selected_score.family,
        selected_kernel=selected_model.kernel_,
        selected_kernel_repr=str(selected_model.kernel_),
        variogram=variogram,
        candidate_scores=candidate_scores,
    )
    return selected_model, result


def _score_kernel_candidate(
    family: str,
    kernel: Any,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    response_range: float,
    gpr_config: GprConfig,
    kernel_prior_config: KernelPriorConfig,
) -> tuple[KernelCandidateScore, GaussianProcessRegressor | None]:
    try:
        predicted, std = _cross_validated_prediction(
            kernel,
            X,
            y,
            gpr_config,
            kernel_prior_config,
        )
        errors = predicted - y
        cv_rmse = float(np.sqrt(np.mean(errors**2)))
        cv_r2 = _safe_r2(y, predicted)
        coverage = float(np.mean(np.abs(errors) <= 1.96 * np.maximum(std, 0.0)))
        coverage_penalty = abs(coverage - kernel_prior_config.coverage_target)
        fitted = _make_gpr(kernel, gpr_config)
        fitted.fit(X, y)
        log_marginal_likelihood = float(
            getattr(fitted, "log_marginal_likelihood_value_", float("nan"))
        )
        score_value = (
            kernel_prior_config.cv_rmse_weight * (cv_rmse / response_range)
            + kernel_prior_config.lml_weight
            * _negative_lml_per_sample(log_marginal_likelihood, len(X))
            + kernel_prior_config.coverage_weight * coverage_penalty
        )
        return (
            KernelCandidateScore(
                family=family,
                kernel_repr=str(fitted.kernel_),
                cv_rmse=cv_rmse,
                cv_r2=cv_r2,
                log_marginal_likelihood=log_marginal_likelihood,
                coverage_95=coverage,
                coverage_penalty=coverage_penalty,
                score=float(score_value),
            ),
            fitted,
        )
    except Exception as exc:
        return (
            KernelCandidateScore(
                family=family,
                kernel_repr=str(kernel),
                cv_rmse=float("nan"),
                cv_r2=float("nan"),
                log_marginal_likelihood=float("nan"),
                coverage_95=float("nan"),
                coverage_penalty=float("nan"),
                score=float("inf"),
                error=str(exc),
            ),
            None,
        )


def _cross_validated_prediction(
    kernel: Any,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    gpr_config: GprConfig,
    kernel_prior_config: KernelPriorConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    predicted = np.empty(len(y), dtype=float)
    std = np.empty(len(y), dtype=float)
    splitter: Iterable[tuple[NDArray[np.int_], NDArray[np.int_]]]
    if len(y) <= 30:
        splitter = LeaveOneOut().split(X)
    else:
        n_splits = min(kernel_prior_config.max_cv_splits, len(y))
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=kernel_prior_config.random_state or gpr_config.random_state,
        ).split(X)
    for train_index, test_index in splitter:
        model = _make_gpr(kernel, gpr_config, optimize_hyperparameters=False)
        model.fit(X[train_index], y[train_index])
        mean, fold_std = model.predict(X[test_index], return_std=True)
        predicted[test_index] = np.asarray(mean, dtype=float).reshape(-1)
        std[test_index] = np.asarray(fold_std, dtype=float).reshape(-1)
    return predicted, std


def _kernel_for_family(family: str, variogram: VariogramSummary) -> Any:
    constant_initial = max(variogram.sill, 1e-6)
    constant_lower = max(constant_initial * 1e-3, 1e-8)
    constant_upper = max(constant_initial * 1e3, constant_lower * 10.0)
    noise_initial = max(min(variogram.noise_level, constant_initial), 1e-10)
    noise_upper = max(constant_initial * 0.75, noise_initial * 10.0, 1e-6)
    constant = ConstantKernel(
        constant_initial,
        constant_value_bounds=(constant_lower, constant_upper),
    )
    white = WhiteKernel(
        noise_level=noise_initial,
        noise_level_bounds=(1e-10, noise_upper),
    )
    length_scale = np.asarray(variogram.length_scale, dtype=float)
    length_bounds = np.asarray(variogram.length_scale_bounds, dtype=float)
    if family == "matern_0.5":
        base = Matern(
            length_scale=length_scale, length_scale_bounds=length_bounds, nu=0.5
        )
    elif family == "matern_1.5":
        base = Matern(
            length_scale=length_scale, length_scale_bounds=length_bounds, nu=1.5
        )
    elif family == "matern_2.5":
        base = Matern(
            length_scale=length_scale, length_scale_bounds=length_bounds, nu=2.5
        )
    elif family == "rbf":
        base = RBF(length_scale=length_scale, length_scale_bounds=length_bounds)
    elif family == "rational_quadratic":
        isotropic_scale = float(np.median(length_scale))
        isotropic_bounds = (
            float(np.min(length_bounds[:, 0])),
            float(np.max(length_bounds[:, 1])),
        )
        base = RationalQuadratic(
            length_scale=isotropic_scale,
            alpha=1.0,
            length_scale_bounds=isotropic_bounds,
            alpha_bounds=(1e-2, 1e2),
        )
    else:
        raise ValueError(f"Unknown kernel family: {family}")
    return constant * base + white


def _make_gpr(
    kernel: Any,
    config: GprConfig,
    optimize_hyperparameters: bool = True,
) -> GaussianProcessRegressor:
    return GaussianProcessRegressor(
        kernel=clone(kernel),
        alpha=config.alpha,
        normalize_y=config.normalize_y,
        optimizer="fmin_l_bfgs_b" if optimize_hyperparameters else None,
        n_restarts_optimizer=(
            config.n_restarts_optimizer if optimize_hyperparameters else 0
        ),
        random_state=config.random_state,
    )


def _domain_span(
    points: NDArray[np.float64],
    bounds: ArrayLike | None,
) -> NDArray[np.float64]:
    if bounds is None:
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
    else:
        bounds_array = np.asarray(bounds, dtype=float)
        if bounds_array.shape != (points.shape[1], 2):
            raise ValueError("bounds must have shape (dimension, 2).")
        lower = bounds_array[:, 0]
        upper = bounds_array[:, 1]
    span = upper - lower
    fallback = np.ptp(points, axis=0)
    span = np.where(span > 1e-12, span, fallback)
    span = np.where(span > 1e-12, span, 1.0)
    return span.astype(float)


def _negative_lml_per_sample(value: float, n_samples: int) -> float:
    if not np.isfinite(value):
        return 1e6
    return float(-value / max(n_samples, 1))


def _safe_r2(y_true: NDArray[np.float64], y_predicted: NDArray[np.float64]) -> float:
    if np.var(y_true) <= 1e-12:
        return float("nan")
    return float(r2_score(y_true, y_predicted))


def _rng(random_state: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)
