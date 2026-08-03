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
from krispu.uncertainty.support import kernel_support_deficit


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
    jackknife_field_sensitivity: NDArray[np.float64] | None
    kernel_support_deficit: NDArray[np.float64] | None
    krispu_uncertainty: NDArray[np.float64] | None
    maximum_kernel_correlation_to_observations: NDArray[np.float64] | None
    jackknife_field_means: NDArray[np.float64] | None
    jackknife_residuals: NDArray[np.float64] | None
    jackknife_standardized_residuals: NDArray[np.float64] | None
    calibrated_posterior_std: NDArray[np.float64] | None
    combined_std: NDArray[np.float64] | None
    jackknife_calibration_factor: float | None
    observed_X: NDArray[np.float64]
    observed_y: NDArray[np.float64]
    observed_jackknife_eligible: NDArray[np.bool_]
    evaluation_points: NDArray[np.float64]
    recommended_point: NDArray[np.float64] | None
    distance_to_nearest_observation: float | None
    jackknife_field_sensitivity_at_selection: float | None
    kernel_support_deficit_at_selection: float | None
    krispu_uncertainty_at_selection: float | None
    maximum_kernel_correlation_at_selection: float | None
    distance_to_domain_boundary: float | None
    near_domain_boundary: bool | None
    on_current_sample_hull: bool | None
    dominant_jackknife_observation_index: int | None
    dominant_jackknife_observation_coordinate: NDArray[np.float64] | None
    dominant_observation_is_anchor: bool | None
    dominant_observation_near_boundary: bool | None
    wall_time_seconds: float
    selection_mode: str = "fixed_generic"
    profile: str | None = None
    selected_kernel_id: str = "matern_32_ard"
    current_length_scales: tuple[float, ...] = ()
    selection_score: float | None = None
    kernel_selection_result: KernelSelectionResult | None = None
    acquisition_field: NDArray[np.float64] | None = None
    acquisition_label: str = "GP posterior standard deviation"

    @property
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
            "mean_posterior_std": mean(self.posterior_std),
            "mean_jackknife_field_sensitivity": mean(self.jackknife_field_sensitivity),
            "mean_kernel_support_deficit": mean(self.kernel_support_deficit),
            "mean_krispu_uncertainty": mean(self.krispu_uncertainty),
            "max_posterior_std": maximum(self.posterior_std),
            "max_jackknife_field_sensitivity": maximum(self.jackknife_field_sensitivity),
            "max_kernel_support_deficit": maximum(self.kernel_support_deficit),
            "max_krispu_uncertainty": maximum(self.krispu_uncertainty),
            "recommended_x": (
                None if self.recommended_point is None else float(self.recommended_point[0])
            ),
            "recommended_y": (
                None if self.recommended_point is None else float(self.recommended_point[1])
            ),
            "nearest_normalized_distance": self.distance_to_nearest_observation,
            "jackknife_field_sensitivity_at_selection": self.jackknife_field_sensitivity_at_selection,
            "kernel_support_deficit_at_selection": self.kernel_support_deficit_at_selection,
            "krispu_uncertainty_at_selection": self.krispu_uncertainty_at_selection,
            "maximum_kernel_correlation_to_observations": (
                self.maximum_kernel_correlation_at_selection
            ),
            "distance_to_domain_boundary": self.distance_to_domain_boundary,
            "near_domain_boundary": self.near_domain_boundary,
            "on_current_sample_hull": self.on_current_sample_hull,
            "dominant_jackknife_observation_index": self.dominant_jackknife_observation_index,
            "dominant_jackknife_observation_x": (
                None
                if self.dominant_jackknife_observation_coordinate is None
                else float(self.dominant_jackknife_observation_coordinate[0])
            ),
            "dominant_jackknife_observation_y": (
                None
                if self.dominant_jackknife_observation_coordinate is None
                else float(self.dominant_jackknife_observation_coordinate[1])
            ),
            "dominant_observation_is_anchor": self.dominant_observation_is_anchor,
            "dominant_observation_near_boundary": self.dominant_observation_near_boundary,
            "wall_time_seconds": self.wall_time_seconds,
            "selection_mode": self.selection_mode,
            "profile": self.profile,
            "selected_kernel_id": self.selected_kernel_id,
            "current_length_scales": ";".join(str(value) for value in self.current_length_scales),
            "selection_score": self.selection_score,
            "mode": self.selection_mode,
            "study": None,
            "acquisition_field": self.acquisition_label,
            "selection_event": (
                self.kernel_selection_result is not None
                and self.kernel_selection_result.selection_evaluated
            ),
            "switch_accepted": (
                None
                if self.kernel_selection_result is None
                else self.kernel_selection_result.switch_accepted
            ),
            "switch_rejection_reason": (
                None
                if self.kernel_selection_result is None
                else self.kernel_selection_result.switch_rejection_reason
            ),
            "parameters_at_bounds": None,
            "selection_runtime": (
                None
                if self.kernel_selection_result is None
                else self.kernel_selection_result.selection_runtime
            ),
            "uncertainty_error_rank_correlation": _rank_correlation(
                self.acquisition_field, self.metrics.absolute_error
            ),
            "high_error_region_capture": _top_region_capture(
                self.acquisition_field, self.metrics.absolute_error
            ),
            "near_neighbor_acquisition_rate": (
                None
                if self.distance_to_nearest_observation is None
                else float(self.distance_to_nearest_observation <= 0.05)
            ),
            "hyperparameters_optimized": (
                None
                if self.kernel_selection_result is None
                else self.kernel_selection_result.optimization_event.hyperparameters_optimized
            ),
            "reselection_triggered": (
                None
                if self.kernel_selection_result is None
                else self.kernel_selection_result.reselection_event.reselection_triggered
            ),
            "reselection_reasons": (
                None
                if self.kernel_selection_result is None
                else "; ".join(self.kernel_selection_result.reselection_event.reselection_reasons)
            ),
            "candidates_evaluated": (
                None
                if self.kernel_selection_result is None
                else "; ".join(self.kernel_selection_result.reselection_event.candidates_evaluated)
            ),
            "previous_kernel_id": (
                None
                if self.kernel_selection_result is None
                else self.kernel_selection_result.previous_kernel_id
            ),
            "challenger_validation_score": (
                None
                if self.kernel_selection_result is None
                else self.kernel_selection_result.reselection_event.challenger_validation_score
            ),
            "score_improvement": (
                None
                if self.kernel_selection_result is None
                else self.kernel_selection_result.reselection_event.score_improvement
            ),
            "length_scale_minimums": (
                None
                if self.kernel_selection_result is None
                else ";".join(
                    str(value)
                    for value in self.kernel_selection_result.optimization_event.length_scale_minimums
                )
            ),
            "length_scale_maximums": (
                None
                if self.kernel_selection_result is None
                else ";".join(
                    str(value)
                    for value in self.kernel_selection_result.optimization_event.length_scale_maximums
                )
            ),
            "reselection_runtime": (
                None
                if self.kernel_selection_result is None
                else self.kernel_selection_result.reselection_event.reselection_runtime
            ),
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
    initial_jackknife_eligible: ArrayLike | None = None,
    minimum_normalized_distance: float = 1.0e-4,
    boundary_margin: float = 0.05,
    kernel_selection_config: KernelSelectionConfig | dict[str, Any] | None = None,
    kernel_schedule: dict[int, str | tuple[str, Any]] | None = None,
    selection_mode_label: str | None = None,
    forced_points: ArrayLike | None = None,
) -> list[SequentialState]:
    """Run one paired-trial method with one fixed candidate pool."""

    method = {
        "krispu_jackknife": "support_adjusted_krispu",
    }.get(method, method)
    supported = {
        "support_adjusted_krispu",
        "raw_jackknife_sensitivity",
        "posterior_std",
        "random",
        "lhs",
        "maximin",
    }
    if method not in supported:
        raise ValueError(f"Unknown sequential method: {method}")
    if minimum_normalized_distance < 0 or boundary_margin < 0:
        raise ValueError("distance and boundary margins must be non-negative.")
    from evaluation.methods.candidate_order import lhs_order, maximin_index, random_order

    initial = domain.validate_points(initial_X, "initial_X")
    pool = domain.validate_points(candidate_pool, "candidate_pool")
    evaluation = domain.validate_points(evaluation_points, "evaluation_points")
    if final_budget < len(initial):
        raise ValueError("final_budget must be at least the initial sample count.")
    if initial_jackknife_eligible is None:
        jackknife_eligible = np.ones(len(initial), dtype=bool)
    else:
        jackknife_eligible = np.asarray(initial_jackknife_eligible)
        if jackknife_eligible.dtype != np.bool_ or jackknife_eligible.shape != (len(initial),):
            raise ValueError(
                "initial_jackknife_eligible must be a Boolean mask matching initial_X."
            )
        jackknife_eligible = jackknife_eligible.copy()
    if not np.any(jackknife_eligible):
        raise ValueError("At least one initial observation must be jackknife-eligible.")
    if len(pool) < final_budget - len(initial):
        raise ValueError("candidate_pool is too small for the requested final budget.")
    forced = (
        None if forced_points is None else domain.validate_points(forced_points, "forced_points")
    )
    if forced is not None and len(forced) != final_budget - len(initial):
        raise ValueError("forced_points must contain one point per adaptive measurement.")
    true_values = (
        np.asarray(hidden_field(evaluation), dtype=float).reshape(-1)
        if true_evaluation is None
        else np.asarray(true_evaluation, dtype=float).reshape(-1)
    )
    if len(true_values) != len(evaluation):
        raise ValueError("true_evaluation must have one value per evaluation point.")
    if metrics_function is None:
        from evaluation.metrics.reconstruction import reconstruction_metrics

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
        observations = ObservationSet(observed_X, observed_y, jackknife_eligible)
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
                scheduled_id, _scheduled_kernel = scheduled
                selection_result = selector.fit_kernel_by_id(
                    scheduled_id, domain.normalize(observed_X), observed_y, gpr_config=config
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
        buffered_jackknife_plan = None
        if selection_result is not None and selection_result.candidate_scores:
            buffered_jackknife_plan = selection_result.candidate_scores[0].fold_plan
        if method in {"support_adjusted_krispu", "raw_jackknife_sensitivity"}:
            recommender = KrispURecommender(
                domain,
                uncertainty=method,
                gpr_config=step_config,
                random_state=random_state,
                n_candidates=len(pool),
                min_normalized_distance=minimum_normalized_distance,
            )
            diagnostics = recommender.evaluate_uncertainty(
                observations, reference, buffered_jackknife_plan=buffered_jackknife_plan
            )
            predicted = diagnostics.predicted_mean
            posterior = diagnostics.posterior_std
            fitted_kernel_for_state = recommender.surrogate_.frozen_kernel
        else:
            surrogate = GPRSurrogate(step_config).fit(domain.normalize(observed_X), observed_y)
            predicted, posterior = surrogate.predict(domain.normalize(reference))
            support_full, correlation_full = kernel_support_deficit(
                surrogate,
                domain.normalize(observed_X),
                domain.normalize(reference),
                epsilon=step_config.response_epsilon,
            )
            fitted_kernel_for_state = surrogate.frozen_kernel

        predicted_full = predicted
        posterior_full = posterior
        predicted = predicted_full[:n_evaluation]
        posterior = posterior_full[:n_evaluation]
        if diagnostics is not None:
            jackknife_sensitivity = diagnostics.jackknife_field_sensitivity[:n_evaluation]
            support_deficit = diagnostics.kernel_support_deficit[:n_evaluation]
            krispu = diagnostics.krispu_uncertainty[:n_evaluation]
            maximum_correlation = diagnostics.maximum_kernel_correlation_to_observations[
                :n_evaluation
            ]
            jackknife_means = diagnostics.jackknife_field_means[:n_evaluation]
            jackknife_residuals = diagnostics.jackknife_residuals.copy()
            jackknife_standardized = diagnostics.jackknife_standardized_residuals.copy()
            candidate_scores = (
                diagnostics.krispu_uncertainty[n_evaluation:]
                if method == "support_adjusted_krispu"
                else diagnostics.jackknife_field_sensitivity[n_evaluation:]
            )
            calibrated = diagnostics.calibrated_posterior_std[:n_evaluation]
            combined = diagnostics.combined_std[:n_evaluation]
            calibration = diagnostics.jackknife_calibration_factor
            candidate_dominant_indices = diagnostics.dominant_jackknife_observation_indices[
                n_evaluation:
            ]
        else:
            jackknife_sensitivity = krispu = None
            support_deficit = support_full[:n_evaluation]
            maximum_correlation = correlation_full[:n_evaluation]
            jackknife_means = jackknife_residuals = jackknife_standardized = None
            candidate_scores = posterior_full[n_evaluation:]
            calibrated = combined = None
            calibration = None
            candidate_dominant_indices = None

        metrics = metrics_function(true_values, predicted)
        next_point = None
        selection = None
        if len(observed_X) < final_budget:
            if forced is not None:
                next_point = forced[len(observed_X) - len(initial)].copy()
                distances = np.linalg.norm(pool - next_point, axis=1)
                selection = int(np.argmin(distances))
                valid = valid_candidate_mask(
                    domain,
                    pool,
                    observed_X,
                    minimum_normalized_distance=minimum_normalized_distance,
                )
                if not valid[selection] or not available[selection]:
                    raise ValueError("forced_points contains an invalid or repeated point.")
            elif method in {
                "support_adjusted_krispu",
                "raw_jackknife_sensitivity",
                "posterior_std",
            }:
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
            if forced is None:
                available[selection] = False
                next_point = pool[selection].copy()
            else:
                available[selection] = False

        selection_values = _selection_diagnostics(
            domain,
            observed_X,
            next_point,
            boundary_margin,
        )
        selected_sensitivity = None
        selected_support = None
        selected_krispu = None
        selected_correlation = None
        if selection is not None:
            candidate_index = n_evaluation + selection
            if diagnostics is not None:
                selected_sensitivity = float(
                    diagnostics.jackknife_field_sensitivity[candidate_index]
                )
                selected_support = float(diagnostics.kernel_support_deficit[candidate_index])
                selected_krispu = float(diagnostics.krispu_uncertainty[candidate_index])
                selected_correlation = float(
                    diagnostics.maximum_kernel_correlation_to_observations[candidate_index]
                )
            else:
                selected_support = float(support_full[candidate_index])
                selected_correlation = float(correlation_full[candidate_index])
        dominant_index = None
        dominant_coordinate = None
        dominant_anchor = None
        dominant_near_boundary = None
        if selection is not None and candidate_dominant_indices is not None:
            dominant_index = int(candidate_dominant_indices[selection])
            dominant_coordinate = observed_X[dominant_index].copy()
            dominant_anchor = not bool(jackknife_eligible[dominant_index])
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
            jackknife_field_sensitivity=(
                None if jackknife_sensitivity is None else jackknife_sensitivity.copy()
            ),
            kernel_support_deficit=(None if support_deficit is None else support_deficit.copy()),
            krispu_uncertainty=None if krispu is None else krispu.copy(),
            maximum_kernel_correlation_to_observations=(
                None if maximum_correlation is None else maximum_correlation.copy()
            ),
            jackknife_field_means=None if jackknife_means is None else jackknife_means.copy(),
            jackknife_residuals=None if jackknife_residuals is None else jackknife_residuals.copy(),
            jackknife_standardized_residuals=(
                None if jackknife_standardized is None else jackknife_standardized.copy()
            ),
            calibrated_posterior_std=None if calibrated is None else calibrated.copy(),
            combined_std=None if combined is None else combined.copy(),
            jackknife_calibration_factor=calibration,
            observed_X=observed_X.copy(),
            observed_y=observed_y.copy(),
            observed_jackknife_eligible=jackknife_eligible.copy(),
            evaluation_points=evaluation.copy(),
            recommended_point=next_point,
            distance_to_nearest_observation=selection_values[0],
            jackknife_field_sensitivity_at_selection=selected_sensitivity,
            kernel_support_deficit_at_selection=selected_support,
            krispu_uncertainty_at_selection=selected_krispu,
            maximum_kernel_correlation_at_selection=selected_correlation,
            distance_to_domain_boundary=selection_values[1],
            near_domain_boundary=selection_values[2],
            on_current_sample_hull=selection_values[3],
            dominant_jackknife_observation_index=dominant_index,
            dominant_jackknife_observation_coordinate=dominant_coordinate,
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
                _kernel_identifier(fitted_kernel_for_state)
                if selection_result is None
                else selection_result.selected_kernel_id
            ),
            current_length_scales=_length_scales_from_kernel(fitted_kernel_for_state),
            selection_score=(
                None if selection_result is None else selection_result.selection_score
            ),
            kernel_selection_result=selection_result,
            acquisition_field=(
                krispu.copy()
                if method == "support_adjusted_krispu" and krispu is not None
                else (
                    jackknife_sensitivity.copy()
                    if method == "raw_jackknife_sensitivity" and jackknife_sensitivity is not None
                    else posterior.copy()
                )
            ),
            acquisition_label=(
                "support-adjusted KRISP-U uncertainty"
                if method == "support_adjusted_krispu"
                else (
                    "raw buffered-jackknife field sensitivity"
                    if method == "raw_jackknife_sensitivity"
                    else "GP posterior standard deviation"
                )
            ),
        )
        states.append(state)
        if next_point is not None:
            observed_X = np.vstack((observed_X, next_point))
            observed_y = np.concatenate(
                (observed_y, np.asarray(hidden_field(next_point), dtype=float).reshape(-1))
            )
            jackknife_eligible = np.concatenate((jackknife_eligible, np.array([True], dtype=bool)))
    return states


def _best_available(
    pool: NDArray[np.float64],
    available: NDArray[np.bool_],
    scores: NDArray[np.float64] | None,
    domain: CandidateDomain,
    observed_X: NDArray[np.float64],
    minimum_normalized_distance: float = 1.0e-4,
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
    direct = kernel.get_params(deep=False).get("length_scale")
    if direct is not None:
        values = np.asarray(direct, dtype=float).reshape(-1)
        return tuple(float(item) for item in values if np.isfinite(item))
    values: list[float] = []
    for name, value in kernel.get_params(deep=True).items():
        if "length_scale" in name:
            array = np.asarray(value, dtype=float).reshape(-1)
            values.extend(float(item) for item in array if np.isfinite(item))
    return tuple(values)


def _kernel_identifier(kernel: Any | None) -> str:
    if kernel is None:
        return "unknown"
    name = type(kernel).__name__.lower()
    return {
        "gaussianard": "gaussian_ard",
        "exponentialard": "exponential_ard",
        "matern32ard": "matern_32_ard",
        "matern52ard": "matern_52_ard",
        "rationalquadraticard": "rational_quadratic_ard",
        "sphericalard": "spherical_ard",
        "wendlandc2ard": "wendland_c2_ard",
    }.get(name, name)


def _rank_correlation(
    first: NDArray[np.float64] | None, second: NDArray[np.float64]
) -> float | None:
    if first is None or len(first) != len(second) or len(first) < 2:
        return None
    first_rank = np.argsort(np.argsort(np.asarray(first, dtype=float)))
    second_rank = np.argsort(np.argsort(np.asarray(second, dtype=float)))
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _top_region_capture(
    uncertainty: NDArray[np.float64] | None,
    error: NDArray[np.float64],
) -> float | None:
    if uncertainty is None or len(uncertainty) != len(error) or len(error) < 2:
        return None
    count = max(1, int(np.ceil(0.10 * len(error))))
    error_region = set(np.argsort(error)[-count:])
    uncertainty_region = set(np.argsort(uncertainty)[-count:])
    return float(len(error_region.intersection(uncertainty_region)) / count)
