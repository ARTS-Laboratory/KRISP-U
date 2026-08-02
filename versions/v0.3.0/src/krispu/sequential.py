"""Small deterministic sequential-design runner used by v0.3.0 audits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import ConvexHull, QhullError

from krispu.candidates import nearest_normalized_distance, valid_candidate_mask
from krispu.config import GPRConfig
from krispu.domains import CandidateDomain
from krispu.kernels.selection import KernelSelectionResult, KernelSelector
from krispu.kernels.specification import KernelSelectionConfig, parse_kernel_configuration
from krispu.observations import ObservationSet
from krispu.recommender import KrispURecommender
from krispu.surrogates.gpr import GPRSurrogate


@dataclass(frozen=True)
class SequentialState:
    field: str
    trial: int
    method: str
    sample_count: int
    initial_sample_count: int
    metrics: Any
    true_field: NDArray[np.float64]
    predicted_field: NDArray[np.float64]
    posterior_std: NDArray[np.float64] | None
    loo_field_uncertainty: NDArray[np.float64] | None
    loo_field_means: NDArray[np.float64] | None
    loo_residuals: NDArray[np.float64] | None
    loo_standardized_residuals: NDArray[np.float64] | None
    calibrated_posterior_std: NDArray[np.float64] | None
    combined_std: NDArray[np.float64] | None
    loo_calibration_factor: float | None
    observed_X: NDArray[np.float64]
    observed_y: NDArray[np.float64]
    observed_loo_eligible: NDArray[np.bool_]
    evaluation_points: NDArray[np.float64]
    recommended_point: NDArray[np.float64] | None
    distance_to_nearest_observation: float | None
    distance_to_domain_boundary: float | None
    near_domain_boundary: bool | None
    on_current_sample_hull: bool | None
    dominant_loo_observation_index: int | None
    dominant_loo_observation_coordinate: NDArray[np.float64] | None
    dominant_observation_is_anchor: bool | None
    dominant_observation_near_boundary: bool | None
    wall_time_seconds: float
    selection_mode: str = "fixed_generic"
    profile: str | None = None
    selected_kernel_id: str = "matern_32_ard"
    current_length_scales: tuple[float, ...] = ()
    selection_score: float | None = None
    kernel_selection_result: KernelSelectionResult | None = None

    @property
    def jackknife_std(self) -> NDArray[np.float64] | None:
        """Compatibility alias for the LOO field uncertainty."""

        return self.loo_field_uncertainty

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
            "mean_loo_field_uncertainty": mean(self.loo_field_uncertainty),
            "mean_calibrated_posterior_std": mean(self.calibrated_posterior_std),
            "max_posterior_std": maximum(self.posterior_std),
            "max_loo_field_uncertainty": maximum(self.loo_field_uncertainty),
            "max_calibrated_posterior_std": maximum(self.calibrated_posterior_std),
            "recommended_x": (
                None if self.recommended_point is None else float(self.recommended_point[0])
            ),
            "recommended_y": (
                None if self.recommended_point is None else float(self.recommended_point[1])
            ),
            "distance_to_nearest_observation": self.distance_to_nearest_observation,
            "distance_to_domain_boundary": self.distance_to_domain_boundary,
            "near_domain_boundary": self.near_domain_boundary,
            "on_current_sample_hull": self.on_current_sample_hull,
            "dominant_loo_observation_index": self.dominant_loo_observation_index,
            "dominant_loo_observation_x": (
                None
                if self.dominant_loo_observation_coordinate is None
                else float(self.dominant_loo_observation_coordinate[0])
            ),
            "dominant_loo_observation_y": (
                None
                if self.dominant_loo_observation_coordinate is None
                else float(self.dominant_loo_observation_coordinate[1])
            ),
            "dominant_observation_is_anchor": self.dominant_observation_is_anchor,
            "dominant_observation_near_boundary": self.dominant_observation_near_boundary,
            "wall_time_seconds": self.wall_time_seconds,
            "selection_mode": self.selection_mode,
            "profile": self.profile,
            "selected_kernel_id": self.selected_kernel_id,
            "current_length_scales": ";".join(str(value) for value in self.current_length_scales),
            "selection_score": self.selection_score,
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
    initial_loo_eligible: ArrayLike | None = None,
    minimum_normalized_distance: float = 0.05,
    boundary_margin: float = 0.05,
    kernel_selection_config: KernelSelectionConfig | dict[str, Any] | None = None,
    kernel_schedule: dict[int, str | tuple[str, Any]] | None = None,
    selection_mode_label: str | None = None,
) -> list[SequentialState]:
    """Run one paired-trial method with one fixed candidate pool."""

    supported = {"krispu_loo", "posterior_std", "random", "lhs", "maximin"}
    if method not in supported:
        raise ValueError(f"Unknown sequential method: {method}")
    if minimum_normalized_distance < 0 or boundary_margin < 0:
        raise ValueError("distance and boundary margins must be non-negative.")
    from benchmarks.methods import lhs_order, maximin_index, random_order

    initial = domain.validate_points(initial_X, "initial_X")
    pool = domain.validate_points(candidate_pool, "candidate_pool")
    evaluation = domain.validate_points(evaluation_points, "evaluation_points")
    if final_budget < len(initial):
        raise ValueError("final_budget must be at least the initial sample count.")
    if initial_loo_eligible is None:
        loo_eligible = np.ones(len(initial), dtype=bool)
    else:
        loo_eligible = np.asarray(initial_loo_eligible)
        if loo_eligible.dtype != np.bool_ or loo_eligible.shape != (len(initial),):
            raise ValueError("initial_loo_eligible must be a Boolean mask matching initial_X.")
        loo_eligible = loo_eligible.copy()
    if not np.any(loo_eligible):
        raise ValueError("At least one initial observation must be LOO-eligible.")
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
    selection_config = (
        None
        if kernel_selection_config is None
        else parse_kernel_configuration(kernel_selection_config)
    )
    selector = (
        None
        if selection_config is None and kernel_schedule is None
        else KernelSelector(selection_config or KernelSelectionConfig(), gpr_config=config)
    )
    available = valid_candidate_mask(
        domain,
        pool,
        observed_X,
        minimum_normalized_distance=minimum_normalized_distance,
    )
    random_indices = random_order(len(pool), random_state)
    lhs_indices = lhs_order(pool, domain, len(pool), random_state)
    states: list[SequentialState] = []

    for _ in range(len(initial), final_budget + 1):
        started = perf_counter()
        observations = ObservationSet(observed_X, observed_y, loo_eligible)
        reference = np.vstack((evaluation, pool))
        n_evaluation = len(evaluation)
        diagnostics = None
        fitted_kernel_for_state: Any | None = None
        selection_result: KernelSelectionResult | None = None
        step_config = config
        if kernel_schedule is not None:
            scheduled = kernel_schedule.get(len(observed_X))
            if scheduled is None:
                raise ValueError(
                    f"kernel_schedule is missing a kernel for sample_count={len(observed_X)}."
                )
            if isinstance(scheduled, tuple):
                scheduled_id, scheduled_kernel = scheduled
                selection_result = KernelSelectionResult(
                    sample_count=len(observed_X),
                    selection_mode="scheduled",
                    profile=None,
                    selected_kernel_id=scheduled_id,
                    previous_kernel_id=None,
                    selection_score=0.0,
                    candidate_scores=(),
                    fitted_kernel=scheduled_kernel,
                    optimized_hyperparameters={},
                    optimizer_restarts=0,
                    selection_runtime=0.0,
                    switch_accepted=True,
                    switch_rejection_reason=None,
                    selection_evaluated=False,
                )
            else:
                assert selector is not None
                selection_result = selector.fit_kernel_by_id(
                    scheduled,
                    domain.normalize(observed_X),
                    observed_y,
                    gpr_config=config,
                )
        elif selector is not None:
            selection_result = selector.select(
                domain.normalize(observed_X), observed_y, gpr_config=config
            )
        if selection_result is not None:
            step_config = replace(
                config,
                kernel=selection_result.fitted_kernel,
                optimize_hyperparameters=False,
            )
        if method == "krispu_loo":
            recommender = KrispURecommender(
                domain,
                uncertainty="krispu_loo",
                gpr_config=step_config,
                random_state=random_state,
                n_candidates=len(pool),
                min_normalized_distance=minimum_normalized_distance,
            )
            diagnostics = recommender.evaluate_uncertainty(observations, reference)
            predicted = diagnostics.predicted_mean
            posterior = diagnostics.posterior_std
            fitted_kernel_for_state = recommender.surrogate_.frozen_kernel
        else:
            surrogate = GPRSurrogate(step_config).fit(domain.normalize(observed_X), observed_y)
            predicted, posterior = surrogate.predict(domain.normalize(reference))
            fitted_kernel_for_state = surrogate.frozen_kernel

        predicted_full = predicted
        posterior_full = posterior
        predicted = predicted_full[:n_evaluation]
        posterior = posterior_full[:n_evaluation]
        if diagnostics is not None:
            loo_uncertainty = diagnostics.loo_field_uncertainty[:n_evaluation]
            loo_means = diagnostics.loo_field_means[:n_evaluation]
            loo_residuals = diagnostics.loo_residuals.copy()
            loo_standardized = diagnostics.loo_standardized_residuals.copy()
            candidate_scores = diagnostics.loo_field_uncertainty[n_evaluation:]
            calibrated = diagnostics.calibrated_posterior_std[:n_evaluation]
            combined = diagnostics.combined_std[:n_evaluation]
            calibration = diagnostics.loo_calibration_factor
            candidate_dominant_indices = diagnostics.dominant_loo_observation_indices[n_evaluation:]
        else:
            loo_uncertainty = loo_means = loo_residuals = loo_standardized = None
            candidate_scores = posterior_full[n_evaluation:]
            calibrated = combined = None
            calibration = None
            candidate_dominant_indices = None

        metrics = metrics_function(true_values, predicted)
        next_point = None
        selection = None
        if len(observed_X) < final_budget:
            if method in {"krispu_loo", "posterior_std"}:
                selection = _best_available(
                    pool,
                    available,
                    candidate_scores,
                    domain,
                    observed_X,
                    minimum_normalized_distance,
                )
            elif method == "random":
                selection = _first_valid_available(
                    random_indices,
                    available,
                    pool,
                    domain,
                    observed_X,
                    minimum_normalized_distance,
                )
            elif method == "lhs":
                selection = _first_valid_available(
                    lhs_indices,
                    available,
                    pool,
                    domain,
                    observed_X,
                    minimum_normalized_distance,
                )
            else:
                selection = maximin_index(
                    pool,
                    observed_X,
                    domain,
                    available,
                    minimum_normalized_distance=minimum_normalized_distance,
                )
            available[selection] = False
            next_point = pool[selection].copy()

        selection_values = _selection_diagnostics(
            domain,
            observed_X,
            next_point,
            boundary_margin,
        )
        dominant_index = None
        dominant_coordinate = None
        dominant_anchor = None
        dominant_near_boundary = None
        if selection is not None and candidate_dominant_indices is not None:
            dominant_index = int(candidate_dominant_indices[selection])
            dominant_coordinate = observed_X[dominant_index].copy()
            dominant_anchor = not bool(loo_eligible[dominant_index])
            dominant_near_boundary = _near_boundary(domain, dominant_coordinate, boundary_margin)
        state = SequentialState(
            field=field_name,
            trial=trial,
            method=method,
            sample_count=len(observed_X),
            initial_sample_count=len(initial),
            metrics=metrics,
            true_field=true_values.copy(),
            predicted_field=predicted.copy(),
            posterior_std=posterior.copy(),
            loo_field_uncertainty=None if loo_uncertainty is None else loo_uncertainty.copy(),
            loo_field_means=None if loo_means is None else loo_means.copy(),
            loo_residuals=None if loo_residuals is None else loo_residuals.copy(),
            loo_standardized_residuals=(
                None if loo_standardized is None else loo_standardized.copy()
            ),
            calibrated_posterior_std=None if calibrated is None else calibrated.copy(),
            combined_std=None if combined is None else combined.copy(),
            loo_calibration_factor=calibration,
            observed_X=observed_X.copy(),
            observed_y=observed_y.copy(),
            observed_loo_eligible=loo_eligible.copy(),
            evaluation_points=evaluation.copy(),
            recommended_point=next_point,
            distance_to_nearest_observation=selection_values[0],
            distance_to_domain_boundary=selection_values[1],
            near_domain_boundary=selection_values[2],
            on_current_sample_hull=selection_values[3],
            dominant_loo_observation_index=dominant_index,
            dominant_loo_observation_coordinate=dominant_coordinate,
            dominant_observation_is_anchor=dominant_anchor,
            dominant_observation_near_boundary=dominant_near_boundary,
            wall_time_seconds=perf_counter() - started,
            selection_mode=(
                selection_mode_label
                if selection_mode_label is not None
                else (
                    "fixed_generic" if selection_result is None else selection_result.selection_mode
                )
            ),
            profile=None if selection_result is None else selection_result.profile,
            selected_kernel_id=(
                "matern_32_ard" if selection_result is None else selection_result.selected_kernel_id
            ),
            current_length_scales=_length_scales_from_kernel(fitted_kernel_for_state),
            selection_score=(
                None if selection_result is None else selection_result.selection_score
            ),
            kernel_selection_result=selection_result,
        )
        states.append(state)
        if next_point is not None:
            observed_X = np.vstack((observed_X, next_point))
            observed_y = np.concatenate(
                (observed_y, np.asarray(hidden_field(next_point), dtype=float).reshape(-1))
            )
            loo_eligible = np.concatenate((loo_eligible, np.array([True], dtype=bool)))
    return states


def _best_available(
    pool: NDArray[np.float64],
    available: NDArray[np.bool_],
    scores: NDArray[np.float64] | None,
    domain: CandidateDomain,
    observed_X: NDArray[np.float64],
    minimum_normalized_distance: float = 0.05,
) -> int:
    """Return the highest-scoring candidate that passes current validity rules."""

    if scores is None:
        raise ValueError("An uncertainty score is required for this method.")
    valid = valid_candidate_mask(
        domain,
        pool,
        observed_X,
        minimum_normalized_distance=minimum_normalized_distance,
    )
    valid &= available & np.isfinite(scores)
    if not np.any(valid):
        raise ValueError("No valid candidate remains.")
    masked = np.where(valid, scores, -np.inf)
    return int(np.argmax(masked))


def _first_valid_available(
    order: NDArray[np.int_],
    available: NDArray[np.bool_],
    pool: NDArray[np.float64],
    domain: CandidateDomain,
    observed_X: NDArray[np.float64],
    minimum_normalized_distance: float,
) -> int:
    valid = valid_candidate_mask(
        domain,
        pool,
        observed_X,
        minimum_normalized_distance=minimum_normalized_distance,
    )
    valid &= available
    for index in order:
        if valid[index]:
            return int(index)
    raise ValueError("No unused candidate remains.")


def _selection_diagnostics(
    domain: CandidateDomain,
    observed_X: NDArray[np.float64],
    selected: NDArray[np.float64] | None,
    boundary_margin: float,
) -> tuple[float | None, float | None, bool | None, bool | None]:
    if selected is None:
        return None, None, None, None
    normalized = domain.normalize(selected.reshape(1, -1))[0]
    distance_to_boundary = float(np.min(np.minimum(normalized, 1.0 - normalized)))
    updated = np.vstack((observed_X, selected))
    on_hull = False
    if updated.shape[1] == 2 and len(updated) >= 3:
        try:
            vertices = ConvexHull(updated).vertices
            on_hull = len(updated) - 1 in vertices
        except QhullError:
            on_hull = False
    return (
        float(nearest_normalized_distance(domain, selected.reshape(1, -1), observed_X)[0]),
        distance_to_boundary,
        distance_to_boundary <= boundary_margin,
        on_hull,
    )


def _near_boundary(domain: CandidateDomain, point: NDArray[np.float64], margin: float) -> bool:
    normalized = domain.normalize(point.reshape(1, -1))[0]
    return bool(np.min(np.minimum(normalized, 1.0 - normalized)) <= margin)


def _length_scales_from_kernel(kernel: Any | None) -> tuple[float, ...]:
    if kernel is None:
        return ()
    values: list[float] = []
    for name, value in kernel.get_params(deep=True).items():
        if "length_scale" in name:
            array = np.asarray(value, dtype=float).reshape(-1)
            values.extend(float(item) for item in array if np.isfinite(item))
    return tuple(values)
