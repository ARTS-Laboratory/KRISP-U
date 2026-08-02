"""Reference fixed-hyperparameter brute-force leave-one-out backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from krispu.observations import ObservationSet
from krispu.surrogates.gpr import GPRSurrogate


@dataclass(frozen=True)
class LOOResult:
    """All candidate/reference-level predictions from eligible LOO folds."""

    reference_points: NDArray[np.float64]
    loo_eligible_indices: NDArray[np.int_]
    field_means: NDArray[np.float64]
    field_stds: NDArray[np.float64]
    heldout_means: NDArray[np.float64]
    heldout_stds: NDArray[np.float64]
    residuals: NDArray[np.float64]
    standardized_residuals: NDArray[np.float64]

    def __post_init__(self) -> None:
        n_reference = len(self.reference_points)
        n_loo = len(self.loo_eligible_indices)
        if self.field_means.shape != (n_reference, n_loo):
            raise ValueError("field_means must have shape (n_reference, n_loo).")
        if self.field_stds.shape != self.field_means.shape:
            raise ValueError("field_stds must match field_means.")
        for value in (
            self.heldout_means,
            self.heldout_stds,
            self.residuals,
            self.standardized_residuals,
        ):
            if value.shape != (n_loo,):
                raise ValueError(
                    "held-out LOO arrays must have one value per eligible observation."
                )


def compute_bruteforce_loo(
    surrogate: GPRSurrogate,
    observations: ObservationSet,
    reference_points_normalized: ArrayLike,
    X_normalized: ArrayLike | None = None,
    epsilon: float | None = None,
) -> LOOResult:
    """Fit every eligible fold with the complete-fit kernel held fixed.

    The returned ``field_means`` matrix is ``(n_reference, n_loo)``.  No
    reduction to a scalar attached to an observed point occurs here.
    """

    if surrogate.model_ is None or surrogate.standardizer_ is None:
        raise ValueError("The complete observation surrogate must be fitted first.")
    reference = _points(reference_points_normalized, "reference_points_normalized")
    points = observations.X if X_normalized is None else _points(X_normalized, "X_normalized")
    if len(points) != len(observations.X) or points.shape[1] != reference.shape[1]:
        raise ValueError("X_normalized must match the observation and reference dimensions.")
    eligible = observations.loo_eligible_indices
    n_reference = len(reference)
    n_loo = len(eligible)
    field_means = np.empty((n_reference, n_loo), dtype=float)
    field_stds = np.empty_like(field_means)
    heldout_means = np.empty(n_loo, dtype=float)
    heldout_stds = np.empty(n_loo, dtype=float)
    residuals = np.empty(n_loo, dtype=float)
    standardized = np.empty(n_loo, dtype=float)
    floor = epsilon if epsilon is not None else surrogate.config.response_epsilon
    if floor <= 0:
        raise ValueError("epsilon must be positive.")

    for column, removed_index in enumerate(eligible):
        keep = np.ones(len(points), dtype=bool)
        keep[removed_index] = False
        if not np.any(keep):
            raise ValueError("LOO requires at least one non-removed observation.")
        fold = GPRSurrogate(surrogate.config).fit_fixed_kernel(
            points[keep],
            observations.y[keep],
            (
                None
                if observations.observation_variances is None
                else observations.observation_variances[keep]
            ),
            standardizer=surrogate.standardizer,
            frozen_kernel=surrogate.frozen_kernel,
        )
        field_mean, field_std = fold.predict(reference)
        heldout_mean, heldout_std = fold.predict(points[removed_index : removed_index + 1])
        if not np.all(np.isfinite(field_mean)) or not np.all(np.isfinite(field_std)):
            raise FloatingPointError(
                f"LOO fold {removed_index} produced a non-finite field prediction."
            )
        if not np.all(np.isfinite(heldout_mean)) or not np.all(np.isfinite(heldout_std)):
            raise FloatingPointError(
                f"LOO fold {removed_index} produced a non-finite held-out prediction."
            )
        residual = float(observations.y[removed_index] - heldout_mean[0])
        denominator = max(float(heldout_std[0]), floor)
        standardized_residual = residual / denominator
        if not np.isfinite(standardized_residual):
            raise FloatingPointError(f"LOO residual for observation {removed_index} is non-finite.")
        field_means[:, column] = field_mean
        field_stds[:, column] = field_std
        heldout_means[column] = heldout_mean[0]
        heldout_stds[column] = heldout_std[0]
        residuals[column] = residual
        standardized[column] = standardized_residual

    return LOOResult(
        reference_points=reference.copy(),
        loo_eligible_indices=eligible.copy(),
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
