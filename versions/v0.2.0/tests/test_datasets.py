from __future__ import annotations

import numpy as np

from krispu.datasets import get_dataset, list_datasets


def test_dataset_catalog_contains_2d_proof_problems() -> None:
    names = set(list_datasets())

    assert "branin" in names
    assert "six_hump_camel" in names
    assert "gaussian_mixture_sparse_candidates" in names


def test_branin_known_optimum_is_close() -> None:
    dataset = get_dataset("branin")
    value = dataset.evaluate(dataset.optimum_x.reshape(1, -1))[0]

    assert np.isclose(value, dataset.optimum_y, atol=1e-3)


def test_discrete_dataset_uses_candidate_space() -> None:
    dataset = get_dataset("branin_irregular_candidates")
    space = dataset.space()
    initial = dataset.initial_design(n=5, random_state=4)

    assert dataset.candidates is not None
    assert np.all(space.contains(initial))
