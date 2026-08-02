"""Domain contracts and physical/normalized coordinate transformations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_points(values: ArrayLike, name: str, dimension: int | None = None) -> NDArray[np.float64]:
    points = np.asarray(values, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array.")
    if dimension is not None and points.shape[1] != dimension:
        raise ValueError(f"{name} must have {dimension} columns.")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values.")
    return points


def _validate_bounds(bounds: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(bounds, dtype=float)
    if result.ndim != 2 or result.shape[1] != 2 or result.shape[0] == 0:
        raise ValueError("bounds must have shape (n_dimensions, 2).")
    if not np.all(np.isfinite(result)):
        raise ValueError("bounds must contain only finite values.")
    if np.any(result[:, 0] >= result[:, 1]):
        raise ValueError("Each lower bound must be less than its upper bound.")
    return result


class Domain(Protocol):
    """Minimal domain interface consumed by the recommender."""

    bounds: NDArray[np.float64]
    dimension: int

    def contains(self, points: ArrayLike) -> NDArray[np.bool_]: ...

    def normalize(self, points: ArrayLike) -> NDArray[np.float64]: ...

    def denormalize(self, points: ArrayLike) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class ContinuousDomain:
    """A rectangular domain with known physical bounds."""

    bounds: NDArray[np.float64] | ArrayLike
    names: tuple[str, ...] | list[str] | None = None

    def __post_init__(self) -> None:
        bounds = _validate_bounds(self.bounds)
        object.__setattr__(self, "bounds", bounds)
        if self.names is not None:
            names = tuple(self.names)
            if len(names) != len(bounds) or len(set(names)) != len(names):
                raise ValueError("names must be unique and match the domain dimension.")
            object.__setattr__(self, "names", names)

    @property
    def dimension(self) -> int:
        return int(self.bounds.shape[0])

    @property
    def lower(self) -> NDArray[np.float64]:
        return self.bounds[:, 0]

    @property
    def upper(self) -> NDArray[np.float64]:
        return self.bounds[:, 1]

    @property
    def span(self) -> NDArray[np.float64]:
        return self.upper - self.lower

    def contains(self, points: ArrayLike, atol: float = 1e-12) -> NDArray[np.bool_]:
        values = _as_points(points, "points", self.dimension)
        return np.all((values >= self.lower - atol) & (values <= self.upper + atol), axis=1)

    def validate_points(self, points: ArrayLike, name: str = "points") -> NDArray[np.float64]:
        values = _as_points(points, name, self.dimension)
        if not np.all(self.contains(values)):
            raise ValueError(f"{name} contains points outside the domain bounds.")
        return values

    def normalize(self, points: ArrayLike) -> NDArray[np.float64]:
        values = self.validate_points(points)
        return (values - self.lower) / self.span

    def denormalize(self, points: ArrayLike) -> NDArray[np.float64]:
        values = _as_points(points, "normalized points", self.dimension)
        if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
            raise ValueError("normalized points must lie in [0, 1].")
        return self.lower + values * self.span

    def normalized_distance(self, first: ArrayLike, second: ArrayLike) -> NDArray[np.float64]:
        left = _as_points(first, "first", self.dimension)
        right = _as_points(second, "second", self.dimension)
        if len(left) != len(right):
            raise ValueError("first and second must have the same number of rows.")
        return np.linalg.norm(self.normalize(left) - self.normalize(right), axis=1)


@dataclass(frozen=True)
class DiscreteCandidateDomain:
    """A finite domain of valid, researcher-supplied candidate coordinates."""

    candidates: NDArray[np.float64] | ArrayLike
    names: tuple[str, ...] | list[str] | None = None

    def __post_init__(self) -> None:
        candidates = _as_points(self.candidates, "candidates")
        rounded = np.round(candidates, 12)
        if len(np.unique(rounded, axis=0)) != len(candidates):
            raise ValueError("candidates must not contain duplicate rows.")
        object.__setattr__(self, "candidates", candidates)
        if self.names is not None:
            names = tuple(self.names)
            if len(names) != candidates.shape[1] or len(set(names)) != len(names):
                raise ValueError("names must be unique and match the domain dimension.")
            object.__setattr__(self, "names", names)

    @property
    def dimension(self) -> int:
        return int(self.candidates.shape[1])

    @property
    def bounds(self) -> NDArray[np.float64]:
        return np.column_stack((np.min(self.candidates, axis=0), np.max(self.candidates, axis=0)))

    def contains(self, points: ArrayLike, atol: float = 1e-10) -> NDArray[np.bool_]:
        values = _as_points(points, "points", self.dimension)
        return np.asarray(
            [np.any(np.all(np.isclose(self.candidates, row, atol=atol), axis=1)) for row in values],
            dtype=bool,
        )

    def validate_points(self, points: ArrayLike, name: str = "points") -> NDArray[np.float64]:
        values = _as_points(points, name, self.dimension)
        if not np.all(self.contains(values)):
            raise ValueError(f"{name} contains a point outside the discrete candidate pool.")
        return values

    def normalize(self, points: ArrayLike) -> NDArray[np.float64]:
        values = self.validate_points(points)
        lower = self.bounds[:, 0]
        span = self.bounds[:, 1] - lower
        span = np.where(span > 0, span, 1.0)
        return (values - lower) / span

    def denormalize(self, points: ArrayLike) -> NDArray[np.float64]:
        values = _as_points(points, "normalized points", self.dimension)
        lower = self.bounds[:, 0]
        span = self.bounds[:, 1] - lower
        span = np.where(span > 0, span, 1.0)
        result = lower + values * span
        return self.validate_points(result, "denormalized points")


@dataclass(frozen=True)
class MixedDomain:
    """Continuous dimensions plus finite numeric options for each discrete axis.

    The discrete columns are only generated from their declared options.  They
    are never optimized through arbitrary intermediate values.
    """

    continuous_bounds: NDArray[np.float64] | ArrayLike
    discrete_options: tuple[tuple[float, ...], ...] | list[list[float]]
    names: tuple[str, ...] | list[str] | None = None

    def __post_init__(self) -> None:
        bounds = _validate_bounds(self.continuous_bounds)
        options = tuple(tuple(float(value) for value in group) for group in self.discrete_options)
        if not options or any(len(group) == 0 for group in options):
            raise ValueError("discrete_options must contain at least one non-empty group.")
        if any(
            len(set(group)) != len(group) or not np.all(np.isfinite(group)) for group in options
        ):
            raise ValueError("discrete options must be finite and unique.")
        object.__setattr__(self, "continuous_bounds", bounds)
        object.__setattr__(self, "discrete_options", options)
        if self.names is not None:
            names = tuple(self.names)
            if len(names) != self.dimension or len(set(names)) != len(names):
                raise ValueError("names must be unique and match the domain dimension.")
            object.__setattr__(self, "names", names)

    @property
    def dimension(self) -> int:
        return int(len(self.continuous_bounds) + len(self.discrete_options))

    @property
    def bounds(self) -> NDArray[np.float64]:
        discrete_bounds = np.asarray([[min(group), max(group)] for group in self.discrete_options])
        return np.vstack((self.continuous_bounds, discrete_bounds))

    def contains(self, points: ArrayLike, atol: float = 1e-10) -> NDArray[np.bool_]:
        values = _as_points(points, "points", self.dimension)
        n_cont = len(self.continuous_bounds)
        continuous_ok = np.all(
            (values[:, :n_cont] >= self.continuous_bounds[:, 0] - atol)
            & (values[:, :n_cont] <= self.continuous_bounds[:, 1] + atol),
            axis=1,
        )
        discrete_ok = np.ones(len(values), dtype=bool)
        for index, options in enumerate(self.discrete_options):
            discrete_ok &= np.any(
                np.isclose(
                    values[:, n_cont + index, None], np.asarray(options)[None, :], atol=atol
                ),
                axis=1,
            )
        return continuous_ok & discrete_ok

    def validate_points(self, points: ArrayLike, name: str = "points") -> NDArray[np.float64]:
        values = _as_points(points, name, self.dimension)
        if not np.all(self.contains(values)):
            raise ValueError(f"{name} contains invalid continuous or discrete coordinates.")
        return values

    def normalize(self, points: ArrayLike) -> NDArray[np.float64]:
        values = self.validate_points(points)
        lower = self.bounds[:, 0]
        span = self.bounds[:, 1] - lower
        span = np.where(span > 0, span, 1.0)
        return (values - lower) / span

    def denormalize(self, points: ArrayLike) -> NDArray[np.float64]:
        values = _as_points(points, "normalized points", self.dimension)
        lower = self.bounds[:, 0]
        span = self.bounds[:, 1] - lower
        span = np.where(span > 0, span, 1.0)
        return self.validate_points(lower + values * span, "denormalized points")

    def all_discrete_combinations(self) -> NDArray[np.float64]:
        return np.asarray(list(product(*self.discrete_options)), dtype=float)


@dataclass(frozen=True)
class PolygonDomain:
    """A two-dimensional polygonal domain, optionally with holes."""

    vertices: NDArray[np.float64] | ArrayLike
    holes: tuple[NDArray[np.float64], ...] | list[ArrayLike] = ()
    names: tuple[str, str] | list[str] | None = None

    def __post_init__(self) -> None:
        vertices = _as_points(self.vertices, "vertices", 2)
        if len(vertices) < 3:
            raise ValueError("A polygon needs at least three vertices.")
        holes = tuple(_as_points(hole, "hole", 2) for hole in self.holes)
        if any(len(hole) < 3 for hole in holes):
            raise ValueError("Each polygon hole needs at least three vertices.")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "holes", holes)
        if self.names is not None:
            names = tuple(self.names)
            if len(names) != 2 or len(set(names)) != 2:
                raise ValueError("names must contain two unique coordinate names.")
            object.__setattr__(self, "names", names)

    @property
    def dimension(self) -> int:
        return 2

    @property
    def bounds(self) -> NDArray[np.float64]:
        return np.column_stack((np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)))

    def contains(self, points: ArrayLike) -> NDArray[np.bool_]:
        values = _as_points(points, "points", 2)
        inside = _in_polygon(values, self.vertices)
        for hole in self.holes:
            inside &= ~_in_polygon(values, hole)
        return inside

    def validate_points(self, points: ArrayLike, name: str = "points") -> NDArray[np.float64]:
        values = _as_points(points, name, 2)
        if not np.all(self.contains(values)):
            raise ValueError(f"{name} contains points outside the polygon domain.")
        return values

    def normalize(self, points: ArrayLike) -> NDArray[np.float64]:
        values = self.validate_points(points)
        lower = self.bounds[:, 0]
        span = self.bounds[:, 1] - lower
        return (values - lower) / span

    def denormalize(self, points: ArrayLike) -> NDArray[np.float64]:
        values = _as_points(points, "normalized points", 2)
        if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
            raise ValueError("normalized points must lie in [0, 1].")
        lower = self.bounds[:, 0]
        span = self.bounds[:, 1] - lower
        return lower + values * span


def _in_polygon(points: NDArray[np.float64], vertices: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Vectorized ray-casting membership test for a closed polygon."""

    x = points[:, 0]
    y = points[:, 1]
    x0 = vertices[:, 0]
    y0 = vertices[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    inside = np.zeros(len(points), dtype=bool)
    for start_x, start_y, end_x, end_y in zip(x0, y0, x1, y1, strict=True):
        crosses = (start_y > y) != (end_y > y)
        denominator = end_y - start_y
        intersections = (end_x - start_x) * (y - start_y) / (
            denominator + (denominator == 0) * 1e-300
        ) + start_x
        inside ^= crosses & (x < intersections)
    return inside


CandidateDomain = ContinuousDomain | DiscreteCandidateDomain | MixedDomain | PolygonDomain
