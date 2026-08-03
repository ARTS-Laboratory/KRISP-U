"""Buffered-jackknife predictive scoring for the finite kernel registry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from krispu.config import GPRConfig
from krispu.jackknife import BufferedJackknifePlan, build_buffered_jackknife_plan
from krispu.kernels.diagnostics import fitted_hyperparameters, inspect_fitted_model
from krispu.kernels.registry import get_kernel_definition
from krispu.observations import ObservationSet
from krispu.surrogates.gpr import GPRSurrogate


@dataclass(frozen=True)
class CandidateScore:
    candidate_kernel_id: str
    display_name: str
    validation_score: float
    buffered_predictive_log_score: float
    upper_tail_normalized_absolute_error: float
    log_marginal_likelihood: float
    degeneracy_penalty: float
    penalty_reasons: tuple[str, ...]
    optimized_hyperparameters: dict[str, list[float]]
    optimizer_restarts: int
    valid: bool
    fitted_kernel: Any | None = None
    fold_plan: BufferedJackknifePlan | None = None

    def as_record(self, sample_count: int, previous_kernel_id: str | None, selected_kernel_id: str, switch_accepted: bool) -> dict[str, Any]:
        return {
            "sample_count": sample_count,
            "candidate_kernel_id": self.candidate_kernel_id,
            "selected_kernel_id": selected_kernel_id,
            "previous_kernel_id": previous_kernel_id,
            "validation_score": self.validation_score,
            "selection_score": self.validation_score,
            "buffered_predictive_log_score": self.buffered_predictive_log_score,
            "upper_tail_normalized_absolute_error": self.upper_tail_normalized_absolute_error,
            "log_marginal_likelihood": self.log_marginal_likelihood,
            "degeneracy_penalty": self.degeneracy_penalty,
            "penalty_reasons": "; ".join(self.penalty_reasons),
            "optimized_hyperparameters": self.optimized_hyperparameters,
            "optimizer_restarts": self.optimizer_restarts,
            "valid": self.valid,
            "switch_accepted": switch_accepted,
        }


def score_candidate_set(
    X: ArrayLike,
    y: ArrayLike,
    candidate_ids: Iterable[str],
    *,
    fold_plan: BufferedJackknifePlan | None = None,
    optimizer_restarts: int = 0,
    random_state: int = 0,
    gpr_config: GPRConfig | None = None,
    warm_start_kernels: dict[str, Any] | None = None,
) -> list[CandidateScore]:
    """Fit and score candidates using one shared buffered fold plan."""

    points = _points(X, "X")
    values = np.asarray(y, dtype=float).reshape(-1)
    if len(points) != len(values) or len(values) < 3 or not np.all(np.isfinite(values)):
        raise ValueError("X and y must be finite and contain at least three observations.")
    base = gpr_config or GPRConfig(random_state=random_state)
    plan = fold_plan or build_buffered_jackknife_plan(
        points,
        multiplier=base.jackknife.multiplier,
        minimum_radius=base.jackknife.minimum_radius,
        maximum_radius=base.jackknife.maximum_radius,
        minimum_training_points=base.jackknife.minimum_training_points,
    )
    observations = ObservationSet(points, values)
    scores: list[CandidateScore] = []
    for index, kernel_id in enumerate(candidate_ids):
        scores.append(
            _score_candidate(
                points,
                values,
                observations,
                plan,
                get_kernel_definition(kernel_id),
                optimizer_restarts=optimizer_restarts,
                random_state=random_state + index,
                gpr_config=base,
                warm_start=None if warm_start_kernels is None else warm_start_kernels.get(kernel_id),
            )
        )
    return scores


def _score_candidate(
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    observations: ObservationSet,
    plan: BufferedJackknifePlan,
    definition: Any,
    *,
    optimizer_restarts: int,
    random_state: int,
    gpr_config: GPRConfig,
    warm_start: Any | None,
) -> CandidateScore:
    try:
        template = warm_start if warm_start is not None else definition.builder(points.shape[1], True)
        config = _fit_config(gpr_config, template, optimizer_restarts, random_state)
        surrogate = GPRSurrogate(config).fit(points, values)
        result = _predictive_metrics(surrogate, observations, points, plan)
        diagnostics = inspect_fitted_model(surrogate.model_, definition)
        reasons = list(diagnostics.reasons)
        valid = diagnostics.valid and np.all(np.isfinite(result))
        if not valid:
            reasons.append("nonfinite validation result")
        return CandidateScore(
            definition.kernel_id,
            definition.display_name,
            float(result[0] + diagnostics.penalty),
            float(result[0]),
            float(result[1]),
            surrogate.log_marginal_likelihood,
            diagnostics.penalty,
            tuple(dict.fromkeys(reasons)),
            fitted_hyperparameters(surrogate.model_),
            optimizer_restarts,
            valid,
            surrogate.frozen_kernel,
            plan,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
        return CandidateScore(
            definition.kernel_id,
            definition.display_name,
            np.inf,
            np.inf,
            np.inf,
            -np.inf,
            100.0,
            (f"fit failure: {type(exc).__name__}: {exc}",),
            {},
            optimizer_restarts,
            False,
            None,
        )


def _predictive_metrics(
    surrogate: GPRSurrogate,
    observations: ObservationSet,
    points: NDArray[np.float64],
    plan: BufferedJackknifePlan,
) -> tuple[float, float]:
    predictions: list[float] = []
    standard: list[float] = []
    truth: list[float] = []
    for anchor, removed in zip(plan.anchor_indices, plan.removed_indices_by_fold, strict=True):
        keep = np.ones(len(points), dtype=bool)
        keep[removed] = False
        fold = GPRSurrogate(surrogate.config).fit_fixed_kernel(
            points[keep], observations.y[keep],
            None if observations.observation_variances is None else observations.observation_variances[keep],
            standardizer=None,
            frozen_kernel=surrogate.frozen_kernel,
        )
        prediction, deviation = fold.predict(points[anchor : anchor + 1])
        predictions.append(float(prediction[0]))
        standard.append(float(deviation[0]))
        truth.append(float(observations.y[anchor]))
    predicted = np.asarray(predictions)
    deviations = np.maximum(np.asarray(standard), surrogate.config.response_epsilon)
    residual = np.asarray(truth) - predicted
    if not np.all(np.isfinite((*predicted, *deviations, *residual))):
        return np.inf, np.inf
    log_score = float(np.mean(0.5 * np.log(2.0 * np.pi * deviations**2) + residual**2 / (2.0 * deviations**2)))
    normalized_absolute = np.abs(residual) / max(float(np.ptp(observations.y)), surrogate.config.response_epsilon)
    return log_score, float(np.quantile(normalized_absolute, 0.90))


def _fit_config(base: GPRConfig, kernel: Any, restarts: int, random_state: int) -> GPRConfig:
    return replace(
        base,
        kernel=kernel,
        optimize_hyperparameters=True,
        n_restarts_optimizer=restarts,
        random_state=random_state,
    )


def _points(value: ArrayLike, name: str) -> NDArray[np.float64]:
    points = np.asarray(value, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] == 0 or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be a finite two-dimensional array.")
    return points
