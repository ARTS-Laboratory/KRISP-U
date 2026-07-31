from __future__ import annotations

import numpy as np

from krispu import KrispUOptimizer, get_dataset
from krispu.models import GprConfig


def test_optimizer_proposes_valid_continuous_point() -> None:
    dataset = get_dataset("quadratic_bowl_2d")
    initial_X = dataset.initial_design(n=5, random_state=5)
    initial_y = dataset.evaluate(initial_X)
    optimizer = KrispUOptimizer(
        dataset.space(),
        objective=dataset.objective,
        random_state=5,
        n_candidates=64,
        optimize_continuous_acquisition=False,
        gpr_config=GprConfig(n_restarts_optimizer=0),
    )

    acquisition = optimizer.ask(initial_X, initial_y)

    assert acquisition.x_next.shape == (2,)
    assert bool(dataset.space().contains(acquisition.x_next)[0])
    assert np.isfinite(acquisition.score)


def test_optimizer_run_returns_history() -> None:
    dataset = get_dataset("quadratic_bowl_2d")
    initial_X = dataset.initial_design(n=5, random_state=6)
    optimizer = KrispUOptimizer(
        dataset.space(),
        objective=dataset.objective,
        random_state=6,
        n_candidates=64,
        optimize_continuous_acquisition=False,
        gpr_config=GprConfig(n_restarts_optimizer=0),
    )

    result = optimizer.run(dataset.evaluate, initial_X, n_iterations=2)

    assert result.X.shape[0] == 7
    assert len(result.acquisitions) == 2
    assert np.all(np.isfinite(result.best_y_history))


def test_optimizer_respects_discrete_candidate_pool() -> None:
    dataset = get_dataset("gaussian_mixture_sparse_candidates")
    initial_X = dataset.initial_design(n=5, random_state=7)
    initial_y = dataset.evaluate(initial_X)
    optimizer = KrispUOptimizer(
        dataset.space(),
        objective=dataset.objective,
        random_state=7,
        n_candidates=64,
        optimize_continuous_acquisition=False,
        gpr_config=GprConfig(n_restarts_optimizer=0),
    )

    acquisition = optimizer.ask(initial_X, initial_y)

    assert bool(dataset.space().contains(acquisition.x_next)[0])


def test_optimizer_kld_acquisition_scores_field_information_gain() -> None:
    dataset = get_dataset("quadratic_bowl_2d")
    initial_X = dataset.initial_design(n=5, random_state=8)
    initial_y = dataset.evaluate(initial_X)
    candidates = dataset.space().dense_grid(points_per_dimension=5)
    optimizer = KrispUOptimizer(
        dataset.space(),
        objective=dataset.objective,
        acquisition="kld",
        random_state=8,
        n_candidates=64,
        optimize_continuous_acquisition=False,
        gpr_config=GprConfig(n_restarts_optimizer=0),
    )

    acquisition = optimizer.ask(
        initial_X,
        initial_y,
        candidates=candidates,
        store_candidates=True,
    )

    assert acquisition.acquisition == "kld"
    assert bool(dataset.space().contains(acquisition.x_next)[0])
    assert acquisition.scores is not None
    assert np.all(np.isfinite(acquisition.scores))
    assert float(np.max(acquisition.scores)) > float(np.min(acquisition.scores))
