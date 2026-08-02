"""Small deterministic sequential-design runner used by v0.3.0 audits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from krispu.candidates import valid_candidate_mask
from krispu.config import GPRConfig
from krispu.domains import CandidateDomain
from krispu.observations import ObservationSet
from krispu.recommender import KrispURecommender
from krispu.surrogates.gpr import GPRSurrogate


@dataclass(frozen=True)
class SequentialState:
    field: str
    trial: int
    method: str
    sample_count: int
    metrics: Any
    true_field: NDArray[np.float64]
    predicted_field: NDArray[np.float64]
    posterior_std: NDArray[np.float64] | None
    jackknife_std: NDArray[np.float64] | None
    calibrated_posterior_std: NDArray[np.float64] | None
    combined_std: NDArray[np.float64] | None
    loo_calibration_factor: float | None
    observed_X: NDArray[np.float64]
    observed_y: NDArray[np.float64]
    evaluation_points: NDArray[np.float64]
    recommended_point: NDArray[np.float64] | None
    wall_time_seconds: float

    def scalar_record(self) -> dict[str, object]:
        def mean(value: NDArray[np.float64] | None) -> float | None:
            return None if value is None else float(np.mean(value))

        def maximum(value: NDArray[np.float64] | None) -> float | None:
            return None if value is None else float(np.max(value))

        return {
            "field": self.field,
            "trial": self.trial,
            "method": self.method,
            "sample_count": self.sample_count,
            "rmse": self.metrics.rmse,
            "nrmse": self.metrics.nrmse,
            "mae": self.metrics.mae,
            "nmae": self.metrics.nmae,
            "r2": self.metrics.r2,
            "p95_absolute_error": self.metrics.p95_absolute_error,
            "max_absolute_error": self.metrics.max_absolute_error,
            "loo_calibration_factor": self.loo_calibration_factor,
            "mean_posterior_std": mean(self.posterior_std),
            "mean_jackknife_std": mean(self.jackknife_std),
            "mean_calibrated_posterior_std": mean(self.calibrated_posterior_std),
            "mean_combined_std": mean(self.combined_std),
            "max_posterior_std": maximum(self.posterior_std),
            "max_jackknife_std": maximum(self.jackknife_std),
            "max_calibrated_posterior_std": maximum(self.calibrated_posterior_std),
            "max_combined_std": maximum(self.combined_std),
            "recommended_x": (
                None if self.recommended_point is None else float(self.recommended_point[0])
            ),
            "recommended_y": (
                None if self.recommended_point is None else float(self.recommended_point[1])
            ),
            "wall_time_seconds": self.wall_time_seconds,
        }


def run_sequential_design(
    hidden_field: Callable[[ArrayLike], NDArray[np.float64]],
    domain: CandidateDomain,
    initial_X: ArrayLike,
    candidate_pool: ArrayLike,
    evaluation_points: ArrayLike,
    method: str,
    final_budget: int,
    random_state: int,
    *,
    field_name: str = "field",
    trial: int = 0,
    true_evaluation: ArrayLike | None = None,
    gpr_config: GPRConfig | None = None,
    metrics_function: Callable[[ArrayLike, ArrayLike], Any] | None = None,
) -> list[SequentialState]:
    """Run one paired-trial method with one fixed candidate pool."""

    supported = {"krispu_combined", "krispu_jackknife", "posterior_std", "random", "lhs", "maximin"}
    if method not in supported:
        raise ValueError(f"Unknown sequential method: {method}")
    from benchmarks.methods import lhs_order, maximin_index, random_order

    initial = domain.validate_points(initial_X, "initial_X")
    pool = domain.validate_points(candidate_pool, "candidate_pool")
    evaluation = domain.validate_points(evaluation_points, "evaluation_points")
    if final_budget < len(initial):
        raise ValueError("final_budget must be at least the initial sample count.")
    if len(pool) < final_budget - len(initial):
        raise ValueError("candidate_pool is too small for the requested final budget.")
    true_values = (
        np.asarray(hidden_field(evaluation), dtype=float).reshape(-1)
        if true_evaluation is None
        else np.asarray(true_evaluation, dtype=float).reshape(-1)
    )
    if len(true_values) != len(evaluation):
        raise ValueError("true_evaluation must have one value per evaluation point.")
    if metrics_function is None:
        from benchmarks.evaluation import reconstruction_metrics

        metrics_function = reconstruction_metrics
    observed_X = initial.copy()
    observed_y = np.asarray(hidden_field(observed_X), dtype=float).reshape(-1)
    if len(observed_y) != len(observed_X):
        raise ValueError("hidden_field must return one value per point.")
    config = gpr_config or GPRConfig(random_state=random_state)
    available = valid_candidate_mask(domain, pool, observed_X)
    random_indices = random_order(len(pool), random_state)
    lhs_indices = lhs_order(
        pool, domain, final_budget - len(initial), random_state, excluded=~available
    )
    states: list[SequentialState] = []

    for _ in range(len(initial), final_budget + 1):
        started = perf_counter()
        observations = ObservationSet(observed_X, observed_y)
        reference = np.vstack((evaluation, pool))
        n_evaluation = len(evaluation)
        diagnostics = None
        if method.startswith("krispu_"):
            recommender = KrispURecommender(
                domain, gpr_config=config, random_state=random_state, n_candidates=len(pool)
            )
            diagnostics = recommender.evaluate_uncertainty(observations, reference)
        else:
            surrogate = GPRSurrogate(config).fit(domain.normalize(observed_X), observed_y)
            predicted, posterior = surrogate.predict(domain.normalize(reference))
        if diagnostics is not None:
            predicted = diagnostics.predicted_mean[:n_evaluation]
            posterior = diagnostics.posterior_std[:n_evaluation]
            candidate_posterior = diagnostics.posterior_std[n_evaluation:]
            jackknife = diagnostics.jackknife_std[:n_evaluation]
            candidate_jackknife = diagnostics.jackknife_std[n_evaluation:]
            calibrated = diagnostics.calibrated_posterior_std[:n_evaluation]
            combined = diagnostics.combined_std[:n_evaluation]
            candidate_combined = diagnostics.combined_std[n_evaluation:]
            calibration = diagnostics.loo_calibration_factor
        else:
            candidate_posterior = posterior[n_evaluation:]
            predicted = predicted[:n_evaluation]
            posterior = posterior[:n_evaluation]
            candidate_jackknife = candidate_combined = None
            jackknife = calibrated = combined = None
            calibration = None
        metrics = metrics_function(true_values, predicted)
        next_point = None
        if len(states) < final_budget - len(initial) + 1 and len(observed_X) < final_budget:
            if method == "krispu_combined":
                scores = candidate_combined
                index = _best_available(pool, available, scores, domain, observed_X)
            elif method == "krispu_jackknife":
                index = _best_available(pool, available, candidate_jackknife, domain, observed_X)
            elif method == "posterior_std":
                index = _best_available(pool, available, candidate_posterior, domain, observed_X)
            elif method == "random":
                index = _first_available(random_indices, available)
            elif method == "lhs":
                lhs_position = len(states)
                candidate_index = int(lhs_indices[lhs_position])
                if not available[candidate_index]:
                    raise RuntimeError("LHS sequence selected a duplicate candidate.")
                index = candidate_index
            else:
                index = maximin_index(pool, observed_X, domain, available)
            available[index] = False
            next_point = pool[index].copy()
        state = SequentialState(
            field=field_name,
            trial=trial,
            method=method,
            sample_count=len(observed_X),
            metrics=metrics,
            true_field=true_values.copy(),
            predicted_field=predicted.copy(),
            posterior_std=posterior.copy(),
            jackknife_std=None if jackknife is None else jackknife.copy(),
            calibrated_posterior_std=None if calibrated is None else calibrated.copy(),
            combined_std=None if combined is None else combined.copy(),
            loo_calibration_factor=calibration,
            observed_X=observed_X.copy(),
            observed_y=observed_y.copy(),
            evaluation_points=evaluation.copy(),
            recommended_point=next_point,
            wall_time_seconds=perf_counter() - started,
        )
        states.append(state)
        if next_point is not None:
            observed_X = np.vstack((observed_X, next_point))
            observed_y = np.concatenate(
                (observed_y, np.asarray(hidden_field(next_point), dtype=float).reshape(-1))
            )
    return states


def _first_available(order: NDArray[np.int_], available: NDArray[np.bool_]) -> int:
    for index in order:
        if available[index]:
            return int(index)
    raise ValueError("No unused candidate remains.")


def _best_available(
    pool: NDArray[np.float64],
    available: NDArray[np.bool_],
    scores: NDArray[np.float64] | None,
    domain: CandidateDomain,
    observed_X: NDArray[np.float64],
) -> int:
    if scores is None:
        raise ValueError("An uncertainty score is required for this method.")
    valid = available & np.isfinite(scores) & domain.contains(pool)
    if not np.any(valid):
        raise ValueError("No valid candidate remains.")
    masked = np.where(valid, scores, -np.inf)
    return int(np.argmax(masked))
