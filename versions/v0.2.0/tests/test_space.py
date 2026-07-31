from __future__ import annotations

import numpy as np
import pytest

from krispu.space import ContinuousSpace, DiscreteCandidateSpace, HybridCandidateSpace


def test_continuous_space_sampling_and_validation() -> None:
    space = ContinuousSpace([[0.0, 1.0], [-1.0, 1.0]], names=("a", "b"))
    samples = space.sample(16, method="lhs", random_state=1)

    assert samples.shape == (16, 2)
    assert np.all(space.contains(samples))
    with pytest.raises(ValueError):
        space.validate_points([[2.0, 0.0]])


def test_discrete_space_rejects_duplicates_and_samples_candidates() -> None:
    candidates = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    space = DiscreteCandidateSpace(candidates)
    sample = space.sample(2, random_state=2)

    assert sample.shape == (2, 2)
    assert np.all(space.contains(sample))
    with pytest.raises(ValueError):
        DiscreteCandidateSpace([[0.0, 0.0], [0.0, 0.0]])


def test_hybrid_space_sampling_and_contains() -> None:
    space = HybridCandidateSpace([[0.0, 1.0]], discrete_options=[[0.0, 1.0, 2.0]])
    samples = space.sample(10, random_state=3)

    assert samples.shape == (10, 2)
    assert np.all(space.contains(samples))
    assert not bool(space.contains([[0.5, 4.0]])[0])
