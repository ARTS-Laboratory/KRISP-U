"""Deterministic field construction and initial-design generation."""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.fields import FIELD_FACTORIES
from krispu.candidates import generate_candidates
from krispu.domains import CandidateDomain, ContinuousDomain


def initial_design(
    name: str,
    domain: CandidateDomain | None = None,
    sample_count: int = 5,
    boundary_margin: float = 0.05,
    random_state: int = 0,
    return_eligibility: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return a deterministic initial design and explicit jackknife mask."""

    if domain is None:
        domain = ContinuousDomain([[-1.0, 1.0], [-1.0, 1.0]])
    if sample_count < 3:
        raise ValueError("sample_count must support a fitted surrogate.")
    if name == "anchored_boundary":
        if domain.dimension != 2 or sample_count != 5:
            raise ValueError("anchored_boundary is only defined for five two-dimensional points.")
        normalized = np.asarray(
            [[0.0, 0.5], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
            dtype=float,
        )
        design = domain.denormalize(normalized)
        eligibility = np.array([False, True, True, True, True])
        return (design, eligibility) if return_eligibility else design
    if name == "clustered_observations":
        center = np.full((sample_count, domain.dimension), 0.5, dtype=float)
        offsets = np.random.default_rng(random_state).normal(
            0.0, 0.008, size=center.shape
        )
        normalized = np.clip(center + offsets, boundary_margin, 1.0 - boundary_margin)
        design = domain.denormalize(normalized)
        eligibility = np.ones(sample_count, dtype=bool)
        return (design, eligibility) if return_eligibility else design
    if name in {"random_interior", "lhs_interior"}:
        candidates = generate_candidates(domain, max(100, sample_count * 50), "lhs", random_state)
        normalized = domain.normalize(candidates)
        interior = candidates[
            np.all((normalized >= boundary_margin) & (normalized <= 1.0 - boundary_margin), axis=1)
        ]
        if len(interior) < sample_count:
            raise ValueError("The interior candidate set is too small for the initial design.")
        selected = (
            np.random.default_rng(random_state).choice(
                len(interior), size=sample_count, replace=False
            )
            if name == "random_interior"
            else np.arange(sample_count)
        )
        design = interior[np.asarray(selected)].copy()
        eligibility = np.ones(sample_count, dtype=bool)
        return (design, eligibility) if return_eligibility else design
    if name != "interior_maximin":
        raise ValueError("Unknown initial design.")
    oversampled = generate_candidates(domain, max(500, sample_count * 100), "lhs", random_state)
    normalized = domain.normalize(oversampled)
    interior = oversampled[
        np.all((normalized >= boundary_margin) & (normalized <= 1.0 - boundary_margin), axis=1)
    ]
    if len(interior) < sample_count:
        raise ValueError("The interior candidate set is too small for the initial design.")
    normalized = domain.normalize(interior)
    selected = [int(np.argmin(np.linalg.norm(normalized - 0.5, axis=1)))]
    while len(selected) < sample_count:
        distances = np.linalg.norm(
            normalized[:, None, :] - normalized[np.asarray(selected)][None, :, :], axis=2
        ).min(axis=1)
        distances[selected] = -np.inf
        selected.append(int(np.argmax(distances)))
    design = interior[np.asarray(selected)].copy()
    eligibility = np.ones(sample_count, dtype=bool)
    return (design, eligibility) if return_eligibility else design


def make_field(field_name: str, seed: int) -> Any:
    factory = FIELD_FACTORIES[field_name]
    try:
        return factory(seed=seed)
    except TypeError:
        return factory()


def regular_grid(domain: Any, size: int) -> np.ndarray:
    if domain.dimension != 2:
        return generate_candidates(domain, max(size * 4, 256), "lhs", 1907 + domain.dimension)
    axes = [np.linspace(lo, hi, size) for lo, hi in domain.bounds]
    mesh = np.meshgrid(*axes, indexing="xy")
    return np.column_stack([item.ravel() for item in mesh])


__all__ = ["initial_design", "make_field", "regular_grid"]
