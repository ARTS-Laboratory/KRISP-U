"""Candidate-space abstractions for KRISP-U."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import qmc


def make_rng(
    random_state: int | np.random.Generator | None = None,
) -> np.random.Generator:
    """Return a NumPy random generator from a seed or existing generator."""

    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def validate_objective(objective: str) -> str:
    """Validate and normalize a response direction."""

    if objective not in {"minimize", "maximize"}:
        raise ValueError("objective must be either 'minimize' or 'maximize'.")
    return objective


def as_2d_float_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Validate an array-like object as a finite 2D float array."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one dimension.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def validate_bounds(bounds: ArrayLike) -> NDArray[np.float64]:
    """Validate finite lower/upper bounds with shape ``(n_dimensions, 2)``."""

    array = np.asarray(bounds, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("bounds must have shape (n_dimensions, 2).")
    if array.shape[0] == 0:
        raise ValueError("bounds must include at least one dimension.")
    if not np.all(np.isfinite(array)):
        raise ValueError("bounds must contain only finite values.")
    if np.any(array[:, 0] >= array[:, 1]):
        raise ValueError("Each lower bound must be less than its upper bound.")
    return array


def ensure_unique_rows(
    values: NDArray[np.float64], name: str, decimals: int = 12
) -> None:
    """Raise if an array contains duplicate rows under rounded comparison."""

    rounded = np.round(values, decimals=decimals)
    if np.unique(rounded, axis=0).shape[0] != values.shape[0]:
        raise ValueError(f"{name} must not contain duplicate rows.")


@dataclass(frozen=True)
class ContinuousSpace:
    """Bounded continuous n-dimensional candidate space."""

    bounds: NDArray[np.float64] | ArrayLike
    names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        bounds = validate_bounds(self.bounds)
        object.__setattr__(self, "bounds", bounds)
        if self.names is not None and len(self.names) != bounds.shape[0]:
            raise ValueError("names must match the number of dimensions.")

    @property
    def dimension(self) -> int:
        return int(self.bounds.shape[0])

    @property
    def lower(self) -> NDArray[np.float64]:
        return self.bounds[:, 0]

    @property
    def upper(self) -> NDArray[np.float64]:
        return self.bounds[:, 1]

    def validate_points(
        self, values: ArrayLike, name: str = "X"
    ) -> NDArray[np.float64]:
        points = as_2d_float_array(values, name)
        if points.shape[1] != self.dimension:
            raise ValueError(f"{name} must have {self.dimension} columns.")
        if not np.all(self.contains(points)):
            raise ValueError(f"{name} contains points outside the continuous bounds.")
        return points

    def contains(self, values: ArrayLike, atol: float = 1e-12) -> NDArray[np.bool_]:
        points = as_2d_float_array(values, "values")
        if points.shape[1] != self.dimension:
            return np.zeros(points.shape[0], dtype=bool)
        return np.all(
            (points >= self.lower - atol) & (points <= self.upper + atol), axis=1
        )

    def sample(
        self,
        n: int,
        method: str = "random",
        random_state: int | np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Sample candidates using random, Latin-hypercube, Sobol, or grid methods."""

        if n <= 0:
            raise ValueError("n must be a positive integer.")
        method = method.lower()
        rng = make_rng(random_state)
        if method in {"random", "uniform"}:
            unit = rng.random((n, self.dimension))
        elif method in {"lhs", "latin_hypercube", "latin-hypercube"}:
            sampler = qmc.LatinHypercube(d=self.dimension, seed=rng)
            unit = sampler.random(n)
        elif method == "sobol":
            sampler = qmc.Sobol(d=self.dimension, scramble=True, seed=rng)
            unit = sampler.random(n)
        elif method in {"grid", "mesh"}:
            points_per_dimension = max(2, int(np.ceil(n ** (1 / self.dimension))))
            return self.dense_grid(points_per_dimension=points_per_dimension)[:n]
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        return qmc.scale(unit, self.lower, self.upper)

    def dense_grid(
        self, points_per_dimension: int = 50, max_points: int = 250_000
    ) -> NDArray[np.float64]:
        """Return a full factorial grid for low-dimensional diagnostics."""

        if points_per_dimension < 2:
            raise ValueError("points_per_dimension must be at least 2.")
        total = points_per_dimension**self.dimension
        if total > max_points:
            raise ValueError(
                "Requested grid is too large; reduce points_per_dimension "
                "or increase max_points explicitly."
            )
        axes = [
            np.linspace(low, high, points_per_dimension) for low, high in self.bounds
        ]
        mesh = np.meshgrid(*axes, indexing="xy")
        return np.column_stack([axis.ravel() for axis in mesh])


@dataclass(frozen=True)
class DiscreteCandidateSpace:
    """Fixed candidate pool for preset experiment/design options."""

    candidates: NDArray[np.float64] | ArrayLike
    names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        candidates = as_2d_float_array(self.candidates, "candidates")
        ensure_unique_rows(candidates, "candidates")
        object.__setattr__(self, "candidates", candidates)
        if self.names is not None and len(self.names) != candidates.shape[1]:
            raise ValueError("names must match the number of dimensions.")

    @property
    def dimension(self) -> int:
        return int(self.candidates.shape[1])

    @property
    def bounds(self) -> NDArray[np.float64]:
        return np.column_stack(
            (np.min(self.candidates, axis=0), np.max(self.candidates, axis=0))
        )

    def validate_points(
        self, values: ArrayLike, name: str = "X"
    ) -> NDArray[np.float64]:
        points = as_2d_float_array(values, name)
        if points.shape[1] != self.dimension:
            raise ValueError(f"{name} must have {self.dimension} columns.")
        if not np.all(self.contains(points)):
            raise ValueError(
                f"{name} contains points outside the preset candidate pool."
            )
        return points

    def contains(self, values: ArrayLike, atol: float = 1e-10) -> NDArray[np.bool_]:
        points = as_2d_float_array(values, "values")
        if points.shape[1] != self.dimension:
            return np.zeros(points.shape[0], dtype=bool)
        matches = []
        for point in points:
            row_matches = np.all(np.isclose(self.candidates, point, atol=atol), axis=1)
            matches.append(bool(np.any(row_matches)))
        return np.asarray(matches, dtype=bool)

    def sample(
        self,
        n: int,
        method: str = "random",
        random_state: int | np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Sample rows from the candidate pool without replacement."""

        if n <= 0:
            raise ValueError("n must be a positive integer.")
        if n > len(self.candidates):
            raise ValueError("n cannot exceed the number of preset candidates.")
        method = method.lower()
        if method in {"first", "grid"}:
            return self.candidates[:n].copy()
        if method not in {"random", "uniform", "lhs", "sobol"}:
            raise ValueError(f"Unknown sampling method for discrete space: {method}")
        rng = make_rng(random_state)
        indices = rng.choice(len(self.candidates), size=n, replace=False)
        return self.candidates[indices].copy()

    def dense_grid(
        self, points_per_dimension: int = 50, max_points: int = 250_000
    ) -> NDArray[np.float64]:
        """Return the preset candidates; arguments are accepted for API symmetry."""

        _ = (points_per_dimension, max_points)
        return self.candidates.copy()


@dataclass(frozen=True)
class HybridCandidateSpace:
    """Mixed space with continuous dimensions followed by encoded discrete options."""

    continuous_bounds: NDArray[np.float64] | ArrayLike
    discrete_options: Sequence[Sequence[float]]
    names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        bounds = validate_bounds(self.continuous_bounds)
        options = tuple(
            tuple(float(value) for value in group) for group in self.discrete_options
        )
        if not options:
            raise ValueError(
                "discrete_options must include at least one discrete dimension."
            )
        for group in options:
            if len(group) == 0:
                raise ValueError(
                    "Each discrete option group must contain at least one value."
                )
            if len(set(group)) != len(group):
                raise ValueError("Discrete option groups must not contain duplicates.")
        object.__setattr__(self, "continuous_bounds", bounds)
        object.__setattr__(self, "discrete_options", options)
        if self.names is not None and len(self.names) != self.dimension:
            raise ValueError("names must match the number of dimensions.")

    @property
    def dimension(self) -> int:
        return int(self.continuous_bounds.shape[0] + len(self.discrete_options))

    @property
    def bounds(self) -> NDArray[np.float64]:
        discrete_bounds = np.asarray(
            [[min(group), max(group)] for group in self.discrete_options], dtype=float
        )
        return np.vstack((self.continuous_bounds, discrete_bounds))

    def validate_points(
        self, values: ArrayLike, name: str = "X"
    ) -> NDArray[np.float64]:
        points = as_2d_float_array(values, name)
        if points.shape[1] != self.dimension:
            raise ValueError(f"{name} must have {self.dimension} columns.")
        if not np.all(self.contains(points)):
            raise ValueError(
                f"{name} contains invalid continuous bounds or discrete options."
            )
        return points

    def contains(self, values: ArrayLike, atol: float = 1e-10) -> NDArray[np.bool_]:
        points = as_2d_float_array(values, "values")
        if points.shape[1] != self.dimension:
            return np.zeros(points.shape[0], dtype=bool)
        n_cont = self.continuous_bounds.shape[0]
        in_bounds = np.all(
            (points[:, :n_cont] >= self.continuous_bounds[:, 0] - atol)
            & (points[:, :n_cont] <= self.continuous_bounds[:, 1] + atol),
            axis=1,
        )
        discrete_ok = np.ones(points.shape[0], dtype=bool)
        for index, group in enumerate(self.discrete_options):
            column = points[:, n_cont + index]
            valid = np.zeros(points.shape[0], dtype=bool)
            for option in group:
                valid |= np.isclose(column, option, atol=atol)
            discrete_ok &= valid
        return in_bounds & discrete_ok

    def sample(
        self,
        n: int,
        method: str = "random",
        random_state: int | np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Sample mixed continuous/discrete candidates."""

        if n <= 0:
            raise ValueError("n must be a positive integer.")
        rng = make_rng(random_state)
        continuous = ContinuousSpace(self.continuous_bounds).sample(n, method, rng)
        discrete_columns = [
            rng.choice(np.asarray(group, dtype=float), size=n, replace=True)
            for group in self.discrete_options
        ]
        return np.column_stack((continuous, *discrete_columns))

    def dense_grid(
        self, points_per_dimension: int = 10, max_points: int = 250_000
    ) -> NDArray[np.float64]:
        """Return a Cartesian grid across continuous axes and discrete options."""

        continuous = ContinuousSpace(self.continuous_bounds).dense_grid(
            points_per_dimension=points_per_dimension,
            max_points=max_points,
        )
        option_rows = np.asarray(list(product(*self.discrete_options)), dtype=float)
        total = len(continuous) * len(option_rows)
        if total > max_points:
            raise ValueError("Requested hybrid grid exceeds max_points.")
        rows = []
        for option_row in option_rows:
            repeated_options = np.repeat(
                option_row.reshape(1, -1), len(continuous), axis=0
            )
            rows.append(np.column_stack((continuous, repeated_options)))
        return np.vstack(rows)
