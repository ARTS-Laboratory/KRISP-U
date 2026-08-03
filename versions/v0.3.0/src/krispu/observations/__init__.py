"""Validated measured coordinates, responses, and jackknife eligibility."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ObservationSet:
    """Current measurements and an explicit mask of removable observations."""

    X: NDArray[np.float64] | ArrayLike
    y: NDArray[np.float64] | ArrayLike
    jackknife_eligible: NDArray[np.bool_] | ArrayLike | None = None
    observation_variances: NDArray[np.float64] | ArrayLike | None = None

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y = np.asarray(self.y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must have shape (n_observations, n_dimensions).")
        if len(X) != len(y) or len(X) < 2:
            raise ValueError("X and y must have the same length and at least two rows.")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain only finite values.")
        if len(np.unique(np.round(X, 12), axis=0)) != len(X):
            raise ValueError("X must not contain duplicate coordinates.")
        if self.jackknife_eligible is None:
            eligible = np.ones(len(X), dtype=bool)
        else:
            eligible_values = np.asarray(self.jackknife_eligible)
            if eligible_values.dtype != np.bool_:
                raise ValueError("jackknife_eligible must be an explicit Boolean mask.")
            eligible = eligible_values.reshape(-1)
        if len(eligible) != len(X):
            raise ValueError("jackknife_eligible must have one Boolean value per observation.")
        if not np.any(eligible):
            raise ValueError("At least one observation must be eligible for jackknife.")
        variances = None
        if self.observation_variances is not None:
            variances = np.asarray(self.observation_variances, dtype=float).reshape(-1)
            if len(variances) != len(X):
                raise ValueError("observation_variances must have one value per observation.")
            if not np.all(np.isfinite(variances)) or np.any(variances < 0):
                raise ValueError("observation_variances must be finite and non-negative.")
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "jackknife_eligible", eligible)
        object.__setattr__(self, "observation_variances", variances)

    @property
    def n_observations(self) -> int:
        return len(self.X)

    def __len__(self) -> int:
        return self.n_observations

    @property
    def dimension(self) -> int:
        return int(self.X.shape[1])

    @property
    def jackknife_eligible_indices(self) -> NDArray[np.int_]:
        return np.flatnonzero(self.jackknife_eligible)

    @property
    def protected_indices(self) -> NDArray[np.int_]:
        return np.flatnonzero(~self.jackknife_eligible)

    def subset(self, indices: ArrayLike) -> ObservationSet:
        selected = np.asarray(indices, dtype=int).reshape(-1)
        if np.any(selected < 0) or np.any(selected >= len(self.X)):
            raise IndexError("observation index is out of range.")
        variances = (
            None if self.observation_variances is None else self.observation_variances[selected]
        )
        return ObservationSet(
            self.X[selected], self.y[selected], self.jackknife_eligible[selected], variances
        )
