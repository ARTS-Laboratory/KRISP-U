"""Deterministic candidate-selection helpers for the audit runner."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from krispu.domains import CandidateDomain

METHODS = (
    "krispu_loo",
    "posterior_std",
    "random",
    "lhs",
    "maximin",
)


def random_order(n_candidates: int, random_state: int) -> NDArray[np.int_]:
    return np.random.default_rng(random_state).permutation(n_candidates)


def lhs_order(
    candidate_pool: NDArray[np.float64],
    domain: CandidateDomain,
    n_points: int,
    random_state: int,
    excluded: NDArray[np.bool_] | None = None,
) -> NDArray[np.int_]:
    """Map one complete LHS sequence to unique nearest candidate rows."""

    design = qmc.LatinHypercube(d=candidate_pool.shape[1], seed=random_state).random(n_points)
    targets = domain.denormalize(design)
    normalized = domain.normalize(candidate_pool)
    selected: list[int] = []
    available = np.ones(len(candidate_pool), dtype=bool)
    if excluded is not None:
        excluded_values = np.asarray(excluded, dtype=bool).reshape(-1)
        if len(excluded_values) != len(candidate_pool):
            raise ValueError("excluded must have one Boolean value per candidate.")
        available &= ~excluded_values
    for target in targets:
        distances = np.linalg.norm(normalized - domain.normalize(target), axis=1)
        distances[~available] = np.inf
        index = int(np.argmin(distances))
        selected.append(index)
        available[index] = False
    return np.asarray(selected, dtype=int)


def maximin_index(
    candidate_pool: NDArray[np.float64],
    observed_X: NDArray[np.float64],
    domain: CandidateDomain,
    available: NDArray[np.bool_],
    minimum_normalized_distance: float = 0.05,
) -> int:
    from krispu.candidates import valid_candidate_mask

    normalized_candidates = domain.normalize(candidate_pool)
    normalized_observed = domain.normalize(observed_X)
    distances = np.linalg.norm(
        normalized_candidates[:, None, :] - normalized_observed[None, :, :], axis=2
    ).min(axis=1)
    valid = valid_candidate_mask(
        domain,
        candidate_pool,
        observed_X,
        minimum_normalized_distance=minimum_normalized_distance,
    )
    distances[~(available & valid)] = -np.inf
    return int(np.argmax(distances))
