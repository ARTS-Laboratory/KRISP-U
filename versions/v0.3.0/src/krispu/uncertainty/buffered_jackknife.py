"""Fixed-complete-fit buffered-jackknife field reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from krispu.jackknife.plan import BufferedJackknifePlan
from krispu.observations import ObservationSet
from krispu.surrogates.gpr import GPRSurrogate


@dataclass(frozen=True)
class BufferedJackknifeResult:
    reference_points: NDArray[np.float64]
    anchor_indices: NDArray[np.int_]
    removed_indices_by_fold: tuple[NDArray[np.int_], ...]
    effective_radius_by_fold: NDArray[np.float64]
    field_means: NDArray[np.float64]
    field_stds: NDArray[np.float64]
    heldout_means: NDArray[np.float64]
    heldout_stds: NDArray[np.float64]
    residuals: NDArray[np.float64]
    standardized_residuals: NDArray[np.float64]

    @property
    def field_predictions(self) -> NDArray[np.float64]:
        return self.field_means


def compute_buffered_jackknife(
    surrogate: GPRSurrogate,
    observations: ObservationSet,
    reference_points_normalized: ArrayLike,
    plan: BufferedJackknifePlan,
    X_normalized: ArrayLike | None = None,
    epsilon: float | None = None,
) -> BufferedJackknifeResult:
    """Fit each fixed-hyperparameter fold after removing its full neighborhood."""

    if surrogate.model_ is None or surrogate.standardizer_ is None:
        raise ValueError("The complete observation surrogate must be fitted first.")
    reference = _points(reference_points_normalized, "reference_points_normalized")
    points = observations.X if X_normalized is None else _points(X_normalized, "X_normalized")
    if points.shape != observations.X.shape or points.shape[1] != reference.shape[1]:
        raise ValueError("X_normalized must match the observation and reference dimensions.")
    if not np.array_equal(plan.anchor_indices, observations.jackknife_eligible_indices):
        raise ValueError("The buffered-jackknife plan does not match the observations.")
    floor = surrogate.config.response_epsilon if epsilon is None else epsilon
    if floor <= 0:
        raise ValueError("epsilon must be positive.")
    n_reference = len(reference)
    n_folds = len(plan.anchor_indices)
    field_means = np.empty((n_reference, n_folds), dtype=float)
    field_stds = np.empty_like(field_means)
    heldout_means = np.empty(n_folds, dtype=float)
    heldout_stds = np.empty(n_folds, dtype=float)
    residuals = np.empty(n_folds, dtype=float)
    standardized = np.empty(n_folds, dtype=float)
    for column, (anchor, removed) in enumerate(zip(plan.anchor_indices, plan.removed_indices_by_fold, strict=True)):
        keep = np.ones(len(points), dtype=bool)
        keep[removed] = False
        if np.count_nonzero(keep) < 1:
            raise ValueError("A buffered-jackknife fold has no training observations.")
        fold = GPRSurrogate(surrogate.config).fit_fixed_kernel(
            points[keep],
            observations.y[keep],
            None if observations.observation_variances is None else observations.observation_variances[keep],
            standardizer=None,
            frozen_kernel=surrogate.frozen_kernel,
        )
        field_mean, field_std = fold.predict(reference)
        heldout_mean, heldout_std = fold.predict(points[anchor : anchor + 1])
        residual = float(observations.y[anchor] - heldout_mean[0])
        standardized_residual = residual / max(float(heldout_std[0]), floor)
        values = (field_mean, field_std, heldout_mean, heldout_std, np.asarray([standardized_residual]))
        if any(not np.all(np.isfinite(value)) for value in values):
            raise FloatingPointError(f"buffered jackknife fold {anchor} produced non-finite values.")
        field_means[:, column] = field_mean
        field_stds[:, column] = field_std
        heldout_means[column] = heldout_mean[0]
        heldout_stds[column] = heldout_std[0]
        residuals[column] = residual
        standardized[column] = standardized_residual
    return BufferedJackknifeResult(
        reference_points=reference.copy(),
        anchor_indices=plan.anchor_indices.copy(),
        removed_indices_by_fold=plan.removed_indices_by_fold,
        effective_radius_by_fold=plan.effective_radius_by_fold.copy(),
        field_means=field_means,
        field_stds=field_stds,
        heldout_means=heldout_means,
        heldout_stds=heldout_stds,
        residuals=residuals,
        standardized_residuals=standardized,
    )


def _points(values: ArrayLike, name: str) -> NDArray[np.float64]:
    points = np.asarray(values, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] == 0 or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be a finite two-dimensional array.")
    return points
