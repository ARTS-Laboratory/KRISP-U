"""Per-step global-kernel optimization with triggered family reselection."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import clone

from krispu.config import GPRConfig
from krispu.jackknife import BufferedJackknifePlan, build_buffered_jackknife_plan
from krispu.kernels.builders import build_kernel_from_spec
from krispu.kernels.diagnostics import fitted_hyperparameters
from krispu.kernels.events import KernelOptimizationEvent, KernelReselectionEvent, KernelSwitchEvent
from krispu.kernels.profiles import KernelProfile, get_profile
from krispu.kernels.registry import candidate_ids, get_kernel_definition
from krispu.kernels.scoring import CandidateScore, score_candidate_set
from krispu.kernels.specification import KernelSelectionConfig, parse_kernel_configuration
from krispu.surrogates.gpr import GPRSurrogate


@dataclass(frozen=True)
class KernelSelectionResult:
    sample_count: int
    selected_kernel_id: str
    previous_kernel_id: str | None
    selection_score: float
    candidate_scores: tuple[CandidateScore, ...]
    fitted_kernel: Any
    optimized_hyperparameters: dict[str, list[float]]
    optimizer_restarts: int
    selection_runtime: float
    switch_accepted: bool
    switch_rejection_reason: str | None
    selection_evaluated: bool
    optimization_event: KernelOptimizationEvent
    reselection_event: KernelReselectionEvent
    switch_event: KernelSwitchEvent
    mode: str = "automatic"
    profile_name: str | None = None

    @property
    def kernel_id(self) -> str:
        return self.selected_kernel_id

    @property
    def selection_mode(self) -> str:
        return self.mode

    @property
    def profile(self) -> str | None:
        return self.profile_name

    @property
    def current_length_scales(self) -> tuple[float, ...]:
        return self.optimized_hyperparameters.get("length_scale", [])

    def candidate_records(self) -> list[dict[str, Any]]:
        return [
            score.as_record(
                self.sample_count,
                self.previous_kernel_id,
                self.selected_kernel_id,
                self.switch_accepted,
            )
            for score in self.candidate_scores
        ]

    def selected_record(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "selection_mode": self.selection_mode,
            "profile": self.profile,
            "selected_kernel_id": self.selected_kernel_id,
            "previous_kernel_id": self.previous_kernel_id,
            "selection_score": self.selection_score,
            "optimized_hyperparameters": self.optimized_hyperparameters,
            "optimizer_restarts": self.optimizer_restarts,
            "selection_runtime": self.selection_runtime,
            "switch_accepted": self.switch_accepted,
            "switch_rejection_reason": self.switch_rejection_reason,
            "selection_evaluated": self.selection_evaluated,
            "reselection_reasons": self.reselection_event.reselection_reasons,
            "hyperparameters_optimized": self.optimization_event.hyperparameters_optimized,
            "reselection_triggered": self.reselection_event.reselection_triggered,
            "candidates_evaluated": self.reselection_event.candidates_evaluated,
            "current_length_scales": self.optimization_event.current_length_scales,
            "length_scale_minimums": self.optimization_event.length_scale_minimums,
            "length_scale_maximums": self.optimization_event.length_scale_maximums,
            "challenger_validation_score": self.reselection_event.challenger_validation_score,
            "best_challenger_kernel_id": self.reselection_event.best_challenger_kernel_id,
            "score_improvement": self.reselection_event.score_improvement,
            "optimization_runtime": self.optimization_event.optimization_runtime,
            "reselection_runtime": self.reselection_event.reselection_runtime,
        }


class KernelSelector:
    def __init__(
        self,
        config: KernelSelectionConfig | dict[str, Any] | None = None,
        *,
        gpr_config: GPRConfig | None = None,
    ) -> None:
        self.config = parse_kernel_configuration(config)
        self.gpr_config = gpr_config or GPRConfig(random_state=self.config.random_state)
        self.profile: KernelProfile | None = (
            get_profile(self.config.profile or "unrestricted_standard")
            if self.config.mode in {"automatic", "hybrid"}
            else None
        )
        self.current_kernel_id: str | None = None
        self.current_kernel: Any | None = None
        self.last_full_check_count: int | None = None
        self.last_current_score: float | None = None
        self.bound_contact_steps = 0
        self.history: list[KernelSelectionResult] = []
        self.optimization_events: list[KernelOptimizationEvent] = []
        self.reselection_events: list[KernelReselectionEvent] = []
        self.switch_events: list[KernelSwitchEvent] = []

    @property
    def allowed_kernel_ids(self) -> tuple[str, ...]:
        if self.config.mode == "manual":
            return ("manual",)
        return (
            candidate_ids(self.config.candidate_set)
            if self.profile is None
            else self.profile.allowed_kernel_ids
        )

    def select(
        self, X: ArrayLike, y: ArrayLike, *, gpr_config: GPRConfig | None = None
    ) -> KernelSelectionResult:
        points = np.asarray(X, dtype=float)
        values = np.asarray(y, dtype=float).reshape(-1)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or len(points) != len(values) or len(values) < 2:
            raise ValueError("X and y must contain matching observations.")
        base = gpr_config or self.gpr_config
        plan = (
            build_buffered_jackknife_plan(
                points,
                multiplier=base.jackknife.multiplier,
                minimum_radius=base.jackknife.minimum_radius,
                maximum_radius=base.jackknife.maximum_radius,
                minimum_training_points=base.jackknife.minimum_training_points,
            )
            if len(points) >= base.jackknife.minimum_training_points + 1
            else None
        )
        if self.config.mode == "manual":
            return self._select_manual(points, values, base)
        kernel_id = self.current_kernel_id or self._default_kernel_id()
        started = perf_counter()
        warm = {kernel_id: self.current_kernel} if self.current_kernel is not None else None
        current = self._score_one(points, values, kernel_id, base, plan, warm)
        optimization_runtime = perf_counter() - started
        length_scales = tuple(current.optimized_hyperparameters.get("length_scale", []))
        minimums, maximums = _scale_bounds(current.fitted_kernel)
        optimization_event = KernelOptimizationEvent(
            len(points),
            kernel_id,
            True,
            length_scales,
            minimums,
            maximums,
            current.validation_score,
            optimization_runtime,
        )
        reasons = self._reselection_reasons(len(points), current)
        reselection_triggered = bool(
            reasons and plan is not None and len(points) >= self.config.reselection.minimum_points
        )
        candidates = (current,)
        reselection_started = perf_counter()
        if reselection_triggered:
            candidates = tuple(
                score_candidate_set(
                    points,
                    values,
                    self.allowed_kernel_ids,
                    fold_plan=plan,
                    optimizer_restarts=self.config.optimizer_restarts,
                    random_state=self.config.random_state,
                    gpr_config=base,
                    warm_start_kernels=warm,
                )
            )
        valid = [
            score for score in candidates if score.valid and np.isfinite(score.validation_score)
        ]
        if not valid:
            raise RuntimeError(
                "No valid global kernel family passed buffered-jackknife validation."
            )
        best = min(
            valid,
            key=lambda score: (
                score.buffered_predictive_log_score,
                score.upper_tail_normalized_absolute_error,
            ),
        )
        previous = self.current_kernel_id
        current_valid = next(
            (score for score in valid if score.candidate_kernel_id == kernel_id), None
        )
        selected = best
        challenger = None if best.candidate_kernel_id == kernel_id else best
        challenger_candidates = [
            score for score in valid if score.candidate_kernel_id != kernel_id
        ]
        best_challenger = min(
            challenger_candidates,
            key=lambda score: (
                score.buffered_predictive_log_score,
                score.upper_tail_normalized_absolute_error,
            ),
            default=None,
        )
        improvement = 0.0
        accepted = False
        reason = None
        if challenger is not None and current_valid is not None:
            improvement = (
                current_valid.buffered_predictive_log_score
                - challenger.buffered_predictive_log_score
            ) / max(abs(current_valid.buffered_predictive_log_score), 1e-12)
            accepted = improvement >= self.config.reselection.minimum_switch_improvement
            if not accepted:
                selected = current_valid
                reason = "challenger improvement below switch threshold"
        elif challenger is not None:
            accepted = True
            improvement = np.inf
        selected_id = selected.candidate_kernel_id
        selected_kernel = clone(selected.fitted_kernel)
        reselection_event = KernelReselectionEvent(
            len(points),
            reselection_triggered,
            tuple(reasons),
            tuple(score.candidate_kernel_id for score in candidates),
            previous,
            selected_id,
            current.validation_score,
            None if best_challenger is None else best_challenger.validation_score,
            float(improvement),
            perf_counter() - reselection_started,
            None if best_challenger is None else best_challenger.candidate_kernel_id,
        )
        accepted_switch = bool(
            accepted
            and previous is not None
            and selected_id != previous
            and reselection_triggered
        )
        switch_event = KernelSwitchEvent(
            len(points),
            previous,
            selected_id,
            accepted_switch,
            current.validation_score,
            None if best_challenger is None else best_challenger.validation_score,
            float(improvement),
        )
        result = KernelSelectionResult(
            len(points),
            selected_id,
            previous,
            selected.validation_score,
            tuple(candidates),
            selected_kernel,
            selected.optimized_hyperparameters,
            self.config.optimizer_restarts,
            optimization_runtime + reselection_event.reselection_runtime,
            accepted_switch,
            reason,
            reselection_triggered,
            optimization_event,
            reselection_event,
            switch_event,
            mode=self.config.mode,
            profile_name=None if self.profile is None else self.profile.name,
        )
        self.current_kernel_id = selected_id
        self.current_kernel = clone(selected_kernel)
        self.last_current_score = selected.validation_score
        if reselection_triggered:
            self.last_full_check_count = len(points)
        self.history.append(result)
        self.optimization_events.append(optimization_event)
        self.reselection_events.append(reselection_event)
        self.switch_events.append(switch_event)
        return result

    def fit_kernel_by_id(
        self, kernel_id: str, X: ArrayLike, y: ArrayLike, *, gpr_config: GPRConfig | None = None
    ) -> KernelSelectionResult:
        self.current_kernel_id = kernel_id
        self.current_kernel = get_kernel_definition(kernel_id).builder(np.asarray(X).shape[1], True)
        return self.select(X, y, gpr_config=gpr_config)

    def _score_one(
        self,
        points: np.ndarray,
        values: np.ndarray,
        kernel_id: str,
        base: GPRConfig,
        plan: BufferedJackknifePlan | None,
        warm: dict[str, Any] | None,
    ) -> CandidateScore:
        if plan is None:
            definition = get_kernel_definition(kernel_id)
            config = GPRConfig(
                **{
                    **base.__dict__,
                    "kernel": definition.builder(points.shape[1], True),
                    "n_restarts_optimizer": self.config.optimizer_restarts,
                }
            )
            fit = GPRSurrogate(config).fit(points, values)
            return CandidateScore(
                kernel_id,
                definition.display_name,
                0.0,
                0.0,
                0.0,
                fit.log_marginal_likelihood,
                0.0,
                ("minimum fit size",),
                fitted_hyperparameters(fit.model_),
                self.config.optimizer_restarts,
                True,
                fit.frozen_kernel,
            )
        return score_candidate_set(
            points,
            values,
            [kernel_id],
            fold_plan=plan,
            optimizer_restarts=self.config.optimizer_restarts,
            random_state=self.config.random_state,
            gpr_config=base,
            warm_start_kernels=warm,
        )[0]

    def _default_kernel_id(self) -> str:
        return self.profile.default_kernel_id if self.profile is not None else "matern_32_ard"

    def _reselection_reasons(self, count: int, current: CandidateScore) -> list[str]:
        reasons: list[str] = []
        if self.current_kernel_id is None and count >= self.config.reselection.minimum_points:
            reasons.append("initial family evaluation")
        if not current.valid:
            reasons.append("fit-failure trigger")
        if any("nonfinite" in reason for reason in current.penalty_reasons):
            reasons.append("nonfinite validation result")
        if any("condition" in reason for reason in current.penalty_reasons):
            reasons.append("numerical conditioning failure")
        if any("length scale at" in reason for reason in current.penalty_reasons):
            self.bound_contact_steps += 1
        else:
            self.bound_contact_steps = 0
        if self.bound_contact_steps >= self.config.reselection.bound_contact_steps:
            reasons.append("bound-contact trigger")
        if (
            self.last_current_score is not None
            and current.validation_score
            > self.last_current_score * (1.0 + self.config.reselection.score_degradation_fraction)
        ):
            reasons.append("score-degradation trigger")
        if (
            self.last_full_check_count is not None
            and count - self.last_full_check_count >= self.config.reselection.maximum_interval
        ):
            reasons.append("maximum-interval trigger")
        return reasons

    def _select_manual(
        self, points: np.ndarray, values: np.ndarray, base: GPRConfig
    ) -> KernelSelectionResult:
        if self.config.specification is None:
            raise ValueError("manual mode requires one single-family kernel specification.")
        template = build_kernel_from_spec(
            self.config.specification, points.shape[1], self.config.optimize_hyperparameters
        )
        fit = GPRSurrogate(
            GPRConfig(
                **{
                    **base.__dict__,
                    "kernel": template,
                    "n_restarts_optimizer": self.config.optimizer_restarts,
                }
            )
        ).fit(points, values)
        scales = tuple(_hyperparameter(fit, "length_scale"))
        minimums, maximums = _scale_bounds(fit.frozen_kernel)
        optimization = KernelOptimizationEvent(
            len(points), "manual", True, scales, minimums, maximums, 0.0, 0.0
        )
        reselection = KernelReselectionEvent(
            len(points),
            False,
            (),
            ("manual",),
            self.current_kernel_id,
            "manual",
            0.0,
            None,
            0.0,
            0.0,
        )
        switch = KernelSwitchEvent(
            len(points), self.current_kernel_id, "manual", True, 0.0, None, 0.0
        )
        result = KernelSelectionResult(
            len(points),
            "manual",
            self.current_kernel_id,
            0.0,
            (),
            fit.frozen_kernel,
            fitted_hyperparameters(fit.model_),
            self.config.optimizer_restarts,
            0.0,
            True,
            None,
            False,
            optimization,
            reselection,
            switch,
            mode="manual",
        )
        self.current_kernel_id, self.current_kernel = "manual", fit.frozen_kernel
        self.history.append(result)
        return result


def _hyperparameter(fit: GPRSurrogate, name: str) -> list[float]:
    return (
        fit.fitted_kernel.get_params(deep=True).get(name, [])
        if fit.model_ is None
        else fit.model_.kernel_.get_params(deep=True).get(name, [])
    )


def _scale_bounds(kernel: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    for hyperparameter in kernel.hyperparameters:
        if hyperparameter.name.endswith("length_scale"):
            values = np.asarray(hyperparameter.bounds, dtype=float)
            return tuple(values[:, 0]), tuple(values[:, 1])
    return (), ()


def select_kernel(
    X: ArrayLike,
    y: ArrayLike,
    config: KernelSelectionConfig | dict[str, Any] | None = None,
    *,
    gpr_config: GPRConfig | None = None,
) -> KernelSelectionResult:
    return KernelSelector(config, gpr_config=gpr_config).select(X, y, gpr_config=gpr_config)
