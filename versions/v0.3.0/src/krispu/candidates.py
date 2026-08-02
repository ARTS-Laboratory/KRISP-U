"""Candidate generation and explicit validity filtering."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import qmc

from krispu.domains import (
    CandidateDomain,
    ContinuousDomain,
    DiscreteCandidateDomain,
    MixedDomain,
    PolygonDomain,
)

DEFAULT_MINIMUM_NORMALIZED_DISTANCE = 1.0e-4


def _rng(random_state: int | np.random.Generator | None) -> np.random.Generator:
    return (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )


def generate_candidates(
    domain: CandidateDomain,
    n: int,
    method: str = "lhs",
    random_state: int | np.random.Generator | None = None,
    max_attempts: int = 100,
) -> NDArray[np.float64]:
    """Generate a finite candidate pool inside ``domain``."""

    if n <= 0:
        raise ValueError("n must be positive.")
    if isinstance(domain, DiscreteCandidateDomain):
        if n > len(domain.candidates):
            raise ValueError("n cannot exceed the number of discrete candidates.")
        indices = _rng(random_state).choice(len(domain.candidates), n, replace=False)
        return domain.candidates[np.sort(indices)].copy()
    if isinstance(domain, MixedDomain):
        method = method.lower()
        if method not in {"random", "uniform", "lhs", "latin_hypercube", "sobol"}:
            raise ValueError("mixed-domain generation supports random, lhs, and sobol.")
        continuous = _generate_unit(n, len(domain.continuous_bounds), method, random_state)
        low = domain.continuous_bounds[:, 0]
        span = domain.continuous_bounds[:, 1] - low
        continuous = low + continuous * span
        rng = _rng(random_state)
        discrete = [
            rng.choice(np.asarray(options), size=n, replace=True)
            for options in domain.discrete_options
        ]
        return np.column_stack((continuous, *discrete))
    if not isinstance(domain, ContinuousDomain):
        raise TypeError("Unsupported candidate domain.")
    method = method.lower()
    if method not in {"random", "uniform", "lhs", "latin_hypercube", "sobol", "grid"}:
        raise ValueError("method must be random, lhs, sobol, or grid.")
    if method == "grid":
        per_axis = max(2, int(np.ceil(n ** (1 / domain.dimension))))
        axes = [np.linspace(lo, hi, per_axis) for lo, hi in domain.bounds]
        mesh = np.meshgrid(*axes, indexing="xy")
        points = np.column_stack([axis.ravel() for axis in mesh])
        return points[:n]
    if not isinstance(domain, PolygonDomain):
        unit = _generate_unit(n, domain.dimension, method, random_state)
        return domain.denormalize(unit)
    rng = _rng(random_state)
    accepted: list[NDArray[np.float64]] = []
    for _ in range(max_attempts):
        batch = domain.denormalize(rng.random((max(n, 64), domain.dimension)))
        batch = batch[domain.contains(batch)]
        accepted.extend(batch)
        if len(accepted) >= n:
            return np.asarray(accepted[:n])
    raise ValueError("Could not generate enough candidates inside the irregular domain.")


def _generate_unit(
    n: int, dimension: int, method: str, random_state: int | np.random.Generator | None
) -> NDArray[np.float64]:
    rng = _rng(random_state)
    if method in {"random", "uniform"}:
        return rng.random((n, dimension))
    if method in {"lhs", "latin_hypercube"}:
        return qmc.LatinHypercube(d=dimension, seed=rng).random(n)
    if method == "sobol":
        return qmc.Sobol(d=dimension, scramble=True, seed=rng).random(n)
    raise ValueError("method must be random, lhs, sobol, or grid.")


def nearest_normalized_distance(
    domain: CandidateDomain, points: ArrayLike, references: ArrayLike
) -> NDArray[np.float64]:
    left = domain.normalize(points)
    right = domain.normalize(references)
    return np.min(np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2), axis=1)


def valid_candidate_mask(
    domain: CandidateDomain,
    candidates: ArrayLike,
    observed: ArrayLike,
    minimum_normalized_distance: float = DEFAULT_MINIMUM_NORMALIZED_DISTANCE,
    excluded_regions: (
        Iterable[Callable[[NDArray[np.float64]], NDArray[np.bool_]]]
        | Callable[[NDArray[np.float64]], NDArray[np.bool_]]
        | None
    ) = None,
) -> NDArray[np.bool_]:
    """Return the conjunction of domain, observed, distance, and exclusion rules."""

    values = np.asarray(candidates, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    observed_values = np.asarray(observed, dtype=float)
    if observed_values.ndim == 1:
        observed_values = observed_values.reshape(1, -1)
    if minimum_normalized_distance < 0:
        raise ValueError("minimum_normalized_distance must be non-negative.")
    mask = domain.contains(values) & np.all(np.isfinite(values), axis=1)
    if len(observed_values) == 0:
        raise ValueError("observed must contain at least one point.")
    exact_observed = np.any(
        np.all(
            np.isclose(
                values[:, None, :], observed_values[None, :, :], atol=1e-10, rtol=0.0
            ),
            axis=2,
        ),
        axis=1,
    )
    mask &= ~exact_observed
    if np.any(mask):
        valid_indices = np.flatnonzero(mask)
        distances = nearest_normalized_distance(domain, values[valid_indices], observed_values)
        mask[valid_indices] &= distances >= minimum_normalized_distance - 1e-12
    for region in _regions(excluded_regions):
        excluded = np.asarray(region(values), dtype=bool).reshape(-1)
        if len(excluded) != len(values):
            raise ValueError("excluded region predicates must return one Boolean per candidate.")
        mask &= ~excluded
    return mask


def _regions(regions: object) -> list[Callable[[NDArray[np.float64]], NDArray[np.bool_]]]:
    if regions is None:
        return []
    if callable(regions):
        return [regions]
    result = list(regions)  # type: ignore[arg-type]
    if not all(callable(region) for region in result):
        raise TypeError("excluded_regions must be callable or an iterable of callables.")
    return result
