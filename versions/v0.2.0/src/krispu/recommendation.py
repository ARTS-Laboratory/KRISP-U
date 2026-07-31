"""One-shot recommendation workflow for researcher-provided datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.gaussian_process import GaussianProcessRegressor

from krispu.models import GprConfig, KernelPriorConfig, KernelPriorResult
from krispu.optimizer import CandidateSpace, KrispUOptimizer
from krispu.space import (
    ContinuousSpace,
    DiscreteCandidateSpace,
    HybridCandidateSpace,
    as_2d_float_array,
    ensure_unique_rows,
    validate_objective,
)


@dataclass
class Recommendation:
    """One proposed next measurement."""

    rank: int
    x: NDArray[np.float64]
    acquisition_score: float
    predicted_mean: float
    predicted_std: float


@dataclass
class RecommendationSet:
    """Ranked one-shot KRISP-U recommendations."""

    recommendations: list[Recommendation]
    objective: str
    acquisition: str
    feature_names: list[str]
    observed_X: NDArray[np.float64]
    observed_y: NDArray[np.float64]
    kernel_prior_result: KernelPriorResult | None = None

    @property
    def best_observed_y(self) -> float:
        if self.objective == "minimize":
            return float(np.min(self.observed_y))
        return float(np.max(self.observed_y))

    @property
    def best_observed_x(self) -> NDArray[np.float64]:
        if self.objective == "minimize":
            index = int(np.argmin(self.observed_y))
        else:
            index = int(np.argmax(self.observed_y))
        return self.observed_X[index].copy()

    def as_array(self) -> NDArray[np.float64]:
        """Return recommendations as rows of feature values."""

        return np.vstack([recommendation.x for recommendation in self.recommendations])

    def to_records(self) -> list[dict[str, float | int]]:
        """Return recommendation rows ready for CSV/JSON output."""

        records: list[dict[str, float | int]] = []
        for recommendation in self.recommendations:
            row: dict[str, float | int] = {
                "rank": recommendation.rank,
                "acquisition_score": recommendation.acquisition_score,
                "predicted_mean": recommendation.predicted_mean,
                "predicted_std": recommendation.predicted_std,
            }
            for name, value in zip(self.feature_names, recommendation.x, strict=True):
                row[name] = float(value)
            records.append(row)
        return records

    @property
    def selected_kernel_family(self) -> str | None:
        if self.kernel_prior_result is None:
            return None
        return self.kernel_prior_result.selected_family

    @property
    def selected_kernel_repr(self) -> str | None:
        if self.kernel_prior_result is None:
            return None
        return self.kernel_prior_result.selected_kernel_repr

    def model_metadata(self) -> dict[str, object] | None:
        """Return selected-kernel metadata for logging or reports."""

        if self.kernel_prior_result is None:
            return None
        return self.kernel_prior_result.to_dict()


def recommend_next(
    X: ArrayLike,
    y: ArrayLike,
    space: CandidateSpace,
    n_recommendations: int = 1,
    objective: str = "minimize",
    acquisition: str = "uncertainty",
    candidates: ArrayLike | None = None,
    n_candidates: int = 4096,
    candidate_method: str = "lhs",
    random_state: int | np.random.Generator | None = None,
    feature_names: list[str] | None = None,
    exclude_observed: bool = True,
    optimize_continuous_acquisition: bool = False,
    model: GaussianProcessRegressor | None = None,
    gpr_config: GprConfig | None = None,
    kernel_prior_config: KernelPriorConfig | None = None,
) -> RecommendationSet:
    """Fit KRISP-U once and return a ranked set of next measurements.

    This is the primary operational workflow for researchers: provide the
    measurements currently in hand, then receive one or more next points to
    measure. The function does not evaluate the response field itself.
    """

    if n_recommendations <= 0:
        raise ValueError("n_recommendations must be a positive integer.")

    objective = validate_objective(objective)
    observed_X, optimizer_space, scored_candidates = _prepare_recommendation_space(
        X, space, candidates
    )
    observed_y = np.asarray(y, dtype=float).reshape(-1)
    if len(observed_X) != len(observed_y):
        raise ValueError("X and y must contain the same number of rows.")
    if feature_names is None:
        feature_names = _space_feature_names(space)
    if len(feature_names) != space.dimension:
        raise ValueError("feature_names must match the number of dimensions.")

    optimizer = KrispUOptimizer(
        optimizer_space,
        objective=objective,
        acquisition=acquisition,
        n_candidates=n_candidates,
        candidate_method=candidate_method,
        random_state=random_state,
        exclude_observed=exclude_observed,
        optimize_continuous_acquisition=optimize_continuous_acquisition,
        model=model,
        gpr_config=gpr_config,
        kernel_prior_config=kernel_prior_config,
    )
    optimizer.fit(observed_X, observed_y)
    acquisition_result = optimizer.ask(
        candidates=scored_candidates,
        n_candidates=n_candidates,
        candidate_method=candidate_method,
        store_candidates=True,
    )
    if (
        acquisition_result.candidates is None
        or acquisition_result.scores is None
        or acquisition_result.predicted_mean is None
        or acquisition_result.predicted_std is None
    ):
        raise RuntimeError("KRISP-U did not return scored candidates.")

    return _rank_recommendations(
        candidates=acquisition_result.candidates,
        scores=acquisition_result.scores,
        predicted_mean=acquisition_result.predicted_mean,
        predicted_std=acquisition_result.predicted_std,
        observed_X=observed_X,
        observed_y=observed_y,
        objective=objective,
        acquisition=acquisition_result.acquisition,
        feature_names=feature_names,
        n_recommendations=n_recommendations,
        kernel_prior_result=optimizer.kernel_prior_result_,
    )


def infer_continuous_space(
    X: ArrayLike,
    feature_names: list[str] | None = None,
    padding_fraction: float = 0.05,
) -> ContinuousSpace:
    """Create a continuous space from observed data ranges with light padding."""

    points = as_2d_float_array(X, "X")
    if padding_fraction < 0:
        raise ValueError("padding_fraction must be non-negative.")
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    span = upper - lower
    zero_span = span == 0
    span[zero_span] = np.maximum(np.abs(lower[zero_span]), 1.0)
    lower = lower - padding_fraction * span
    upper = upper + padding_fraction * span
    return ContinuousSpace(np.column_stack((lower, upper)), names=feature_names)


def _prepare_recommendation_space(
    X: ArrayLike,
    space: CandidateSpace,
    candidates: ArrayLike | None,
) -> tuple[NDArray[np.float64], CandidateSpace, NDArray[np.float64] | None]:
    if isinstance(space, DiscreteCandidateSpace):
        observed_X = as_2d_float_array(X, "X")
        if observed_X.shape[1] != space.dimension:
            raise ValueError(f"X must have {space.dimension} columns.")
        scored_candidates = (
            space.candidates.copy()
            if candidates is None
            else as_2d_float_array(candidates, "candidates")
        )
        if scored_candidates.shape[1] != space.dimension:
            raise ValueError(f"candidates must have {space.dimension} columns.")
        ensure_unique_rows(scored_candidates, "candidates")
        union = np.unique(
            np.round(np.vstack((space.candidates, observed_X, scored_candidates)), 12),
            axis=0,
        )
        optimizer_space = DiscreteCandidateSpace(union, names=space.names)
        return observed_X, optimizer_space, scored_candidates
    if isinstance(space, ContinuousSpace | HybridCandidateSpace):
        return (
            space.validate_points(X, "X"),
            space,
            (
                None
                if candidates is None
                else space.validate_points(candidates, "candidates")
            ),
        )
    raise TypeError("space must be a KRISP-U candidate space.")


def _rank_recommendations(
    candidates: NDArray[np.float64],
    scores: NDArray[np.float64],
    predicted_mean: NDArray[np.float64],
    predicted_std: NDArray[np.float64],
    observed_X: NDArray[np.float64],
    observed_y: NDArray[np.float64],
    objective: str,
    acquisition: str,
    feature_names: list[str],
    n_recommendations: int,
    kernel_prior_result: KernelPriorResult | None,
) -> RecommendationSet:
    finite_mask = (
        np.isfinite(scores) & np.isfinite(predicted_mean) & np.isfinite(predicted_std)
    )
    if not np.any(finite_mask):
        raise ValueError("No finite acquisition scores were produced.")

    candidates = candidates[finite_mask]
    scores = scores[finite_mask]
    predicted_mean = predicted_mean[finite_mask]
    predicted_std = predicted_std[finite_mask]
    n_recommendations = min(n_recommendations, len(candidates))
    order = np.argsort(scores)[::-1][:n_recommendations]

    recommendations = [
        Recommendation(
            rank=rank,
            x=candidates[index].copy(),
            acquisition_score=float(scores[index]),
            predicted_mean=float(predicted_mean[index]),
            predicted_std=float(predicted_std[index]),
        )
        for rank, index in enumerate(order, start=1)
    ]
    return RecommendationSet(
        recommendations=recommendations,
        objective=objective,
        acquisition=acquisition,
        feature_names=feature_names,
        observed_X=observed_X.copy(),
        observed_y=observed_y.copy(),
        kernel_prior_result=kernel_prior_result,
    )


def _space_feature_names(space: CandidateSpace) -> list[str]:
    names = getattr(space, "names", None)
    if names is not None:
        return list(names)
    if isinstance(space, DiscreteCandidateSpace):
        return [f"x{index + 1}" for index in range(space.dimension)]
    return [f"x{index + 1}" for index in range(space.dimension)]
