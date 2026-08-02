"""Hysteretic kernel-family selection independent of acquisition magnitude."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import clone

from krispu.config import GPRConfig
from krispu.kernels.builders import build_kernel_from_spec
from krispu.kernels.diagnostics import fitted_hyperparameters
from krispu.kernels.profiles import KernelProfile, get_profile
from krispu.kernels.registry import KernelDefinition, candidate_ids, get_kernel_definition
from krispu.kernels.scoring import CandidateScore, score_candidate_set
from krispu.kernels.specification import KernelSelectionConfig, parse_kernel_configuration
from krispu.surrogates.gpr import GPRSurrogate


@dataclass(frozen=True)
class KernelSelectionResult:
    sample_count: int
    selection_mode: str
    profile: str | None
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

    @property
    def kernel_id(self) -> str:
        return self.selected_kernel_id

    def candidate_records(self) -> list[dict[str, Any]]:
        return [
            score.as_record(
                self.sample_count,
                self.selection_mode,
                self.profile,
                self.previous_kernel_id,
                self.selected_kernel_id,
                self.switch_accepted,
                self.switch_rejection_reason,
                self.selection_runtime,
            )
            for score in self.candidate_scores
        ]

    def selected_record(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "selection_mode": self.selection_mode,
            "profile": self.profile,
            "candidate_kernel_id": self.selected_kernel_id,
            "selected_kernel_id": self.selected_kernel_id,
            "previous_kernel_id": self.previous_kernel_id,
            "selection_score": self.selection_score,
            "optimized_hyperparameters": self.optimized_hyperparameters,
            "optimizer_restarts": self.optimizer_restarts,
            "selection_runtime": self.selection_runtime,
            "switch_accepted": self.switch_accepted,
            "switch_rejection_reason": self.switch_rejection_reason,
            "selection_evaluated": self.selection_evaluated,
        }


class KernelSelector:
    """Stateful selector implementing minimum-data and switching hysteresis."""

    def __init__(
        self,
        config: KernelSelectionConfig | dict[str, Any] | None = None,
        *,
        gpr_config: GPRConfig | None = None,
    ) -> None:
        self.config = parse_kernel_configuration(config)
        self.gpr_config = gpr_config or GPRConfig(random_state=self.config.random_state)
        self.profile: KernelProfile | None = None
        if self.config.mode == "hybrid":
            self.profile = get_profile(self.config.profile)
        elif self.config.mode == "automatic":
            self.profile = get_profile(
                self.config.profile if self.config.profile else "unrestricted_standard"
            )
        self.current_kernel_id: str | None = None
        self.last_evaluation_count: int | None = None
        self.history: list[KernelSelectionResult] = []

    @property
    def allowed_kernel_ids(self) -> tuple[str, ...]:
        if self.config.mode == "manual":
            return ("manual",)
        if self.config.mode == "automatic":
            return candidate_ids(self.config.candidate_set)
        assert self.profile is not None
        return self.profile.allowed_kernel_ids

    def select(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        gpr_config: GPRConfig | None = None,
    ) -> KernelSelectionResult:
        """Select and fit a family using predictive scores only."""

        points = np.asarray(X, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        values = np.asarray(y, dtype=float).reshape(-1)
        if points.ndim != 2 or len(points) != len(values) or len(values) < 2:
            raise ValueError("X and y must contain matching observations.")
        if self.config.mode == "manual":
            return self._select_manual(points, values, gpr_config)
        if (
            self.current_kernel_id is not None
            and self.last_evaluation_count is not None
            and len(points) - self.last_evaluation_count < self.config.reevaluate_every
        ):
            return self._refit_current(points, values, gpr_config)
        if len(points) < self.config.minimum_points_before_selection:
            definition = self._default_definition()
            fitted, _ = self._fit_default(definition, points, values, gpr_config)
            result = KernelSelectionResult(
                sample_count=len(points),
                selection_mode=self.config.mode,
                profile=None if self.profile is None else self.profile.name,
                selected_kernel_id=definition.kernel_id,
                previous_kernel_id=self.current_kernel_id,
                selection_score=0.0,
                candidate_scores=(),
                fitted_kernel=fitted.frozen_kernel,
                optimized_hyperparameters=fitted_hyperparameters(fitted.model_),
                optimizer_restarts=self.config.optimizer_restarts,
                selection_runtime=0.0,
                switch_accepted=self.current_kernel_id != definition.kernel_id,
                switch_rejection_reason="minimum_points_before_selection",
                selection_evaluated=False,
            )
            self.current_kernel_id = definition.kernel_id
            self.history.append(result)
            return result

        started = perf_counter()
        base = gpr_config or self.gpr_config
        scores = score_candidate_set(
            points,
            values,
            self.allowed_kernel_ids,
            selection_metric=self.config.selection_metric,
            optimizer_restarts=self.config.optimizer_restarts,
            random_state=self.config.random_state,
            spatial_folds=self.config.spatial_folds,
            nlpd_weight=self.config.nlpd_weight,
            nrmse_weight=self.config.nrmse_weight,
            calibration_weight=self.config.calibration_weight,
            gpr_config=base,
        )
        valid = [score for score in scores if score.valid and np.isfinite(score.selection_score)]
        if not valid:
            raise RuntimeError("All permitted kernel candidates failed predictive validation.")
        best = min(valid, key=lambda score: score.selection_score)
        previous = self.current_kernel_id
        selected = best
        switch_accepted = previous is None or previous == best.candidate_kernel_id
        rejection_reason: str | None = None
        if previous is not None and previous != best.candidate_kernel_id:
            current = next(
                (score for score in valid if score.candidate_kernel_id == previous), None
            )
            if current is None:
                switch_accepted = True
                rejection_reason = "previous kernel was invalid"
            elif (
                current.selection_score - best.selection_score
                >= self.config.minimum_score_improvement
            ):
                switch_accepted = True
            else:
                switch_accepted = False
                selected = current
                rejection_reason = "challenger improvement below hysteresis threshold"
        selected_kernel = selected.fitted_kernel
        if selected_kernel is None:
            raise RuntimeError("A selected candidate did not return a fitted kernel.")
        result = KernelSelectionResult(
            sample_count=len(points),
            selection_mode=self.config.mode,
            profile=None if self.profile is None else self.profile.name,
            selected_kernel_id=selected.candidate_kernel_id,
            previous_kernel_id=previous,
            selection_score=float(selected.selection_score),
            candidate_scores=tuple(scores),
            fitted_kernel=clone(selected_kernel),
            optimized_hyperparameters=selected.optimized_hyperparameters,
            optimizer_restarts=self.config.optimizer_restarts,
            selection_runtime=perf_counter() - started,
            switch_accepted=switch_accepted,
            switch_rejection_reason=rejection_reason,
            selection_evaluated=True,
        )
        self.current_kernel_id = result.selected_kernel_id
        self.last_evaluation_count = len(points)
        self.history.append(result)
        return result

    def fit_kernel_by_id(
        self,
        kernel_id: str,
        X: ArrayLike,
        y: ArrayLike,
        *,
        gpr_config: GPRConfig | None = None,
    ) -> KernelSelectionResult:
        """Fit a registry family for acquisition-isolation replay."""

        definition = get_kernel_definition(kernel_id)
        points = np.asarray(X, dtype=float)
        values = np.asarray(y, dtype=float).reshape(-1)
        base = gpr_config or self.gpr_config
        template = definition.builder(points.shape[1], True)
        fitted = GPRSurrogate(
            replace(
                base,
                kernel=template,
                optimize_hyperparameters=True,
                n_restarts_optimizer=self.config.optimizer_restarts,
            )
        ).fit(points, values)
        result = KernelSelectionResult(
            sample_count=len(points),
            selection_mode=self.config.mode,
            profile=None if self.profile is None else self.profile.name,
            selected_kernel_id=kernel_id,
            previous_kernel_id=self.current_kernel_id,
            selection_score=0.0,
            candidate_scores=(),
            fitted_kernel=fitted.frozen_kernel,
            optimized_hyperparameters=fitted_hyperparameters(fitted.model_),
            optimizer_restarts=self.config.optimizer_restarts,
            selection_runtime=0.0,
            switch_accepted=True,
            switch_rejection_reason=None,
            selection_evaluated=False,
        )
        self.current_kernel_id = kernel_id
        return result

    def _select_manual(
        self,
        points: np.ndarray,
        values: np.ndarray,
        gpr_config: GPRConfig | None,
    ) -> KernelSelectionResult:
        if self.config.specification is None:
            raise ValueError("manual mode requires a kernel specification.")
        started = perf_counter()
        base = gpr_config or self.gpr_config
        template = build_kernel_from_spec(
            self.config.specification,
            points.shape[1],
            self.config.optimize_hyperparameters,
        )
        fitted = GPRSurrogate(
            replace(
                base,
                kernel=template,
                optimize_hyperparameters=self.config.optimize_hyperparameters,
                n_restarts_optimizer=self.config.optimizer_restarts,
            )
        ).fit(points, values)
        result = KernelSelectionResult(
            sample_count=len(points),
            selection_mode="manual",
            profile=None,
            selected_kernel_id="manual",
            previous_kernel_id=self.current_kernel_id,
            selection_score=0.0,
            candidate_scores=(),
            fitted_kernel=fitted.frozen_kernel,
            optimized_hyperparameters=fitted_hyperparameters(fitted.model_),
            optimizer_restarts=self.config.optimizer_restarts,
            selection_runtime=perf_counter() - started,
            switch_accepted=True,
            switch_rejection_reason=None,
            selection_evaluated=False,
        )
        self.current_kernel_id = "manual"
        self.history.append(result)
        return result

    def _refit_current(
        self,
        points: np.ndarray,
        values: np.ndarray,
        gpr_config: GPRConfig | None,
    ) -> KernelSelectionResult:
        assert self.current_kernel_id is not None
        definition = (
            self._default_definition()
            if self.current_kernel_id == "manual"
            else get_kernel_definition(self.current_kernel_id)
        )
        fitted, _ = self._fit_default(definition, points, values, gpr_config)
        result = KernelSelectionResult(
            sample_count=len(points),
            selection_mode=self.config.mode,
            profile=None if self.profile is None else self.profile.name,
            selected_kernel_id=self.current_kernel_id,
            previous_kernel_id=self.current_kernel_id,
            selection_score=0.0,
            candidate_scores=(),
            fitted_kernel=fitted.frozen_kernel,
            optimized_hyperparameters=fitted_hyperparameters(fitted.model_),
            optimizer_restarts=self.config.optimizer_restarts,
            selection_runtime=0.0,
            switch_accepted=False,
            switch_rejection_reason="reevaluation hysteresis hold",
            selection_evaluated=False,
        )
        self.history.append(result)
        return result

    def _default_definition(self) -> KernelDefinition:
        if self.profile is None:
            return get_kernel_definition("matern_32_ard")
        return get_kernel_definition(self.profile.default_kernel_id)

    def _fit_default(
        self,
        definition: KernelDefinition,
        points: np.ndarray,
        values: np.ndarray,
        gpr_config: GPRConfig | None,
    ) -> tuple[GPRSurrogate, CandidateScore]:
        base = gpr_config or self.gpr_config
        template = definition.builder(points.shape[1], True)
        fitted = GPRSurrogate(
            replace(
                base,
                kernel=template,
                optimize_hyperparameters=True,
                n_restarts_optimizer=self.config.optimizer_restarts,
            )
        ).fit(points, values)
        score = CandidateScore(
            candidate_kernel_id=definition.kernel_id,
            display_name=definition.display_name,
            selection_score=float("nan"),
            loo_nrmse=float("nan"),
            loo_nlpd=float("nan"),
            loo_mae=float("nan"),
            loo_calibration_error=float("nan"),
            spatial_cv_nrmse=float("nan"),
            spatial_cv_nlpd=float("nan"),
            spatial_cv_mae=float("nan"),
            spatial_cv_calibration_error=float("nan"),
            log_marginal_likelihood=fitted.log_marginal_likelihood,
            degeneracy_penalty=0.0,
            penalty_reasons=("minimum_points_before_selection",),
            optimized_hyperparameters=fitted_hyperparameters(fitted.model_),
            optimizer_restarts=self.config.optimizer_restarts,
            valid=True,
            fitted_kernel=fitted.frozen_kernel,
        )
        return fitted, score


def select_kernel(
    X: ArrayLike,
    y: ArrayLike,
    config: KernelSelectionConfig | dict[str, Any],
    *,
    gpr_config: GPRConfig | None = None,
) -> KernelSelectionResult:
    """Convenience one-shot selection function."""

    return KernelSelector(config, gpr_config=gpr_config).select(X, y, gpr_config=gpr_config)
