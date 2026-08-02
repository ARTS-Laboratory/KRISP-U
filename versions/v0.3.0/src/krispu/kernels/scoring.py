"""Predictive cross-validation scores for the finite kernel registry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from krispu.config import GPRConfig
from krispu.kernels.diagnostics import (
    fitted_hyperparameters,
    inspect_fitted_model,
)
from krispu.kernels.registry import KernelDefinition, get_kernel_definition
from krispu.surrogates.gpr import GPRSurrogate


@dataclass(frozen=True)
class CandidateScore:
    candidate_kernel_id: str
    display_name: str
    selection_score: float
    loo_nrmse: float
    loo_nlpd: float
    loo_mae: float
    loo_calibration_error: float
    spatial_cv_nrmse: float
    spatial_cv_nlpd: float
    spatial_cv_mae: float
    spatial_cv_calibration_error: float
    log_marginal_likelihood: float
    degeneracy_penalty: float
    penalty_reasons: tuple[str, ...]
    optimized_hyperparameters: dict[str, list[float]]
    optimizer_restarts: int
    valid: bool
    spatial_cv_fallback: bool = False
    fitted_kernel: Any | None = None

    def as_record(
        self,
        sample_count: int,
        selection_mode: str,
        profile: str | None,
        previous_kernel_id: str | None,
        selected_kernel_id: str,
        switch_accepted: bool,
        switch_rejection_reason: str | None,
        selection_runtime: float,
    ) -> dict[str, Any]:
        return {
            "sample_count": sample_count,
            "selection_mode": selection_mode,
            "profile": profile,
            "candidate_kernel_id": self.candidate_kernel_id,
            "selected_kernel_id": selected_kernel_id,
            "previous_kernel_id": previous_kernel_id,
            "selection_score": self.selection_score,
            "loo_nrmse": self.loo_nrmse,
            "loo_nlpd": self.loo_nlpd,
            "spatial_cv_nrmse": self.spatial_cv_nrmse,
            "spatial_cv_nlpd": self.spatial_cv_nlpd,
            "log_marginal_likelihood": self.log_marginal_likelihood,
            "degeneracy_penalty": self.degeneracy_penalty,
            "penalty_reasons": "; ".join(self.penalty_reasons),
            "optimized_hyperparameters": self.optimized_hyperparameters,
            "optimizer_restarts": self.optimizer_restarts,
            "selection_runtime": selection_runtime,
            "switch_accepted": switch_accepted,
            "switch_rejection_reason": switch_rejection_reason,
            "spatial_cv_fallback": self.spatial_cv_fallback,
        }


def spatial_block_folds(
    coordinates: ArrayLike,
    requested_folds: int = 4,
) -> tuple[NDArray[np.int_], ...]:
    """Create deterministic spatially separated folds from normalized points.

    Two-dimensional observations are assigned to occupied quadrants.  In one
    dimension, sorted intervals are used.  Fewer than four observations (or a
    single occupied block) returns an empty tuple, signalling an LOO fallback.
    """

    points = np.asarray(coordinates, dtype=float)
    if points.ndim != 2 or points.shape[0] < 4 or not np.all(np.isfinite(points)):
        return ()
    requested = min(int(requested_folds), len(points))
    if requested < 2:
        return ()
    if points.shape[1] == 1:
        order = np.argsort(points[:, 0], kind="stable")
        labels = np.empty(len(points), dtype=int)
        labels[order] = np.arange(len(points)) * requested // len(points)
    else:
        medians = np.median(points[:, :2], axis=0)
        if requested == 2:
            labels = (points[:, 0] >= medians[0]).astype(int)
        else:
            labels = (points[:, 0] >= medians[0]).astype(int) + 2 * (
                points[:, 1] >= medians[1]
            ).astype(int)
            if requested < 4:
                labels %= requested
    folds = tuple(np.flatnonzero(labels == label) for label in sorted(set(labels)))
    return folds if len(folds) >= 2 and all(len(fold) >= 1 for fold in folds) else ()


def score_candidate_set(
    X: ArrayLike,
    y: ArrayLike,
    candidate_ids: Iterable[str],
    *,
    selection_metric: str = "spatial_cv_composite",
    optimizer_restarts: int = 0,
    random_state: int = 0,
    spatial_folds: int = 4,
    nlpd_weight: float = 0.5,
    nrmse_weight: float = 0.4,
    calibration_weight: float = 0.1,
    gpr_config: GPRConfig | None = None,
) -> list[CandidateScore]:
    """Fit every requested candidate and assign a lower-is-better score."""

    points = _points(X, "X")
    values = np.asarray(y, dtype=float).reshape(-1)
    if len(points) != len(values) or len(values) < 3 or not np.all(np.isfinite(values)):
        raise ValueError("X and y must be finite and contain at least three observations.")
    if selection_metric not in {"loo_predictive", "spatial_block_cv", "spatial_cv_composite"}:
        raise ValueError("Unknown kernel selection metric.")
    base = gpr_config or GPRConfig(random_state=random_state)
    raw: list[CandidateScore] = []
    for index, kernel_id in enumerate(candidate_ids):
        definition = get_kernel_definition(kernel_id)
        raw.append(
            _score_candidate(
                points,
                values,
                definition,
                selection_metric=selection_metric,
                optimizer_restarts=optimizer_restarts,
                random_state=random_state + index,
                spatial_folds=spatial_folds,
                gpr_config=base,
            )
        )
    _assign_scores(raw, selection_metric, nlpd_weight, nrmse_weight, calibration_weight)
    return raw


def _score_candidate(
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    definition: KernelDefinition,
    *,
    selection_metric: str,
    optimizer_restarts: int,
    random_state: int,
    spatial_folds: int,
    gpr_config: GPRConfig,
) -> CandidateScore:
    try:
        template = definition.builder(points.shape[1], True)
        config = replace(
            gpr_config,
            kernel=template,
            optimize_hyperparameters=True,
            n_restarts_optimizer=optimizer_restarts,
            random_state=random_state,
        )
        surrogate = GPRSurrogate(config).fit(points, values)
        degeneracy = inspect_fitted_model(surrogate.model_, definition)
        loo_folds = tuple(np.asarray([index], dtype=int) for index in range(len(points)))
        spatial = spatial_block_folds(points, spatial_folds)
        spatial_fallback = not spatial
        spatial_folds_used = spatial if spatial else loo_folds
        loo_metrics = _evaluate_folds(surrogate, points, values, loo_folds)
        spatial_metrics = _evaluate_folds(surrogate, points, values, spatial_folds_used)
        reasons = list(degeneracy.reasons)
        if spatial_fallback:
            reasons.append("spatial block CV unavailable; fell back to LOO")
        if not np.all(np.isfinite((*loo_metrics, *spatial_metrics))):
            reasons.append("nonfinite CV predictions")
        valid = degeneracy.valid and "nonfinite CV predictions" not in reasons
        penalty = degeneracy.penalty
        if not valid:
            penalty += 100.0
        return CandidateScore(
            candidate_kernel_id=definition.kernel_id,
            display_name=definition.display_name,
            selection_score=np.inf,
            loo_nrmse=loo_metrics[0],
            loo_nlpd=loo_metrics[1],
            loo_mae=loo_metrics[2],
            loo_calibration_error=loo_metrics[3],
            spatial_cv_nrmse=spatial_metrics[0],
            spatial_cv_nlpd=spatial_metrics[1],
            spatial_cv_mae=spatial_metrics[2],
            spatial_cv_calibration_error=spatial_metrics[3],
            log_marginal_likelihood=surrogate.log_marginal_likelihood,
            degeneracy_penalty=penalty,
            penalty_reasons=tuple(dict.fromkeys(reasons)),
            optimized_hyperparameters=fitted_hyperparameters(surrogate.model_),
            optimizer_restarts=optimizer_restarts,
            valid=valid,
            spatial_cv_fallback=spatial_fallback,
            fitted_kernel=surrogate.frozen_kernel,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
        reason = (
            "failed covariance factorization"
            if isinstance(exc, np.linalg.LinAlgError)
            else f"fit failure: {type(exc).__name__}: {exc}"
        )
        return CandidateScore(
            candidate_kernel_id=definition.kernel_id,
            display_name=definition.display_name,
            selection_score=np.inf,
            loo_nrmse=np.inf,
            loo_nlpd=np.inf,
            loo_mae=np.inf,
            loo_calibration_error=np.inf,
            spatial_cv_nrmse=np.inf,
            spatial_cv_nlpd=np.inf,
            spatial_cv_mae=np.inf,
            spatial_cv_calibration_error=np.inf,
            log_marginal_likelihood=-np.inf,
            degeneracy_penalty=100.0,
            penalty_reasons=(reason,),
            optimized_hyperparameters={},
            optimizer_restarts=optimizer_restarts,
            valid=False,
        )


def _evaluate_folds(
    surrogate: GPRSurrogate,
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    test_folds: tuple[NDArray[np.int_], ...],
) -> tuple[float, float, float, float]:
    predictions: list[float] = []
    standard_deviations: list[float] = []
    truth: list[float] = []
    for test_indices in test_folds:
        keep = np.ones(len(points), dtype=bool)
        keep[test_indices] = False
        if not np.any(keep):
            raise ValueError("A cross-validation fold has no training observations.")
        fold = GPRSurrogate(surrogate.config).fit_fixed_kernel(
            points[keep],
            values[keep],
            standardizer=surrogate.standardizer,
            frozen_kernel=surrogate.frozen_kernel,
        )
        prediction, standard_deviation = fold.predict(points[test_indices])
        predictions.extend(prediction.tolist())
        standard_deviations.extend(standard_deviation.tolist())
        truth.extend(values[test_indices].tolist())
    predicted = np.asarray(predictions, dtype=float)
    standard = np.asarray(standard_deviations, dtype=float)
    observed = np.asarray(truth, dtype=float)
    if (
        not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(standard))
        or np.any(standard <= 0)
    ):
        return (np.inf, np.inf, np.inf, np.inf)
    residual = observed - predicted
    nrmse = float(np.sqrt(np.mean(residual**2)) / max(np.ptp(observed), 1e-12))
    mae = float(np.mean(np.abs(residual)))
    nlpd = float(
        np.mean(0.5 * np.log(2.0 * np.pi * standard**2) + residual**2 / (2.0 * standard**2))
    )
    standardized = residual / standard
    calibration = float(abs(np.mean(standardized**2) - 1.0))
    return nrmse, nlpd, mae, calibration


def _assign_scores(
    scores: list[CandidateScore],
    selection_metric: str,
    nlpd_weight: float,
    nrmse_weight: float,
    calibration_weight: float,
) -> None:
    valid = [
        (index, score)
        for index, score in enumerate(scores)
        if score.valid and np.isfinite(score.log_marginal_likelihood)
    ]
    if not valid:
        return
    if selection_metric == "loo_predictive":
        for score_index, score in valid:
            scores[score_index] = replace(
                score,
                selection_score=score.loo_nrmse
                + score.loo_nlpd
                + score.loo_calibration_error
                + score.degeneracy_penalty,
            )
        return
    if selection_metric == "spatial_block_cv":
        for score_index, score in valid:
            scores[score_index] = replace(
                score,
                selection_score=score.spatial_cv_nrmse
                + score.spatial_cv_nlpd
                + score.spatial_cv_calibration_error
                + score.degeneracy_penalty,
            )
        return
    nlpds = _normalize([score.spatial_cv_nlpd for _, score in valid])
    nrmse = _normalize([score.spatial_cv_nrmse for _, score in valid])
    calibration = _normalize([score.spatial_cv_calibration_error for _, score in valid])
    for (score_index, score), value_nlpd, value_nrmse, value_cal in zip(
        valid, nlpds, nrmse, calibration, strict=True
    ):
        scores[score_index] = replace(
            score,
            selection_score=(
                nlpd_weight * value_nlpd
                + nrmse_weight * value_nrmse
                + calibration_weight * value_cal
                + score.degeneracy_penalty
            ),
        )


def _normalize(values: list[float]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    low, high = float(np.min(array)), float(np.max(array))
    return np.zeros_like(array) if np.isclose(low, high) else (array - low) / (high - low)


def _points(value: ArrayLike, name: str) -> NDArray[np.float64]:
    points = np.asarray(value, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] == 0 or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be a finite two-dimensional array.")
    return points
