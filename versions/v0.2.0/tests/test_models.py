from __future__ import annotations

import numpy as np
from sklearn.gaussian_process.kernels import RBF

from krispu import ContinuousSpace, KrispUOptimizer, get_dataset, recommend_next
from krispu.models import (
    GprConfig,
    KernelPriorConfig,
    estimate_variogram_summary,
    fit_prior_optimized_gpr,
    make_kernel_candidates,
)


def test_empirical_variogram_summary_is_finite() -> None:
    dataset = get_dataset("quadratic_bowl_2d")
    X = dataset.initial_design(n=6, random_state=21)
    y = dataset.evaluate(X)

    summary = estimate_variogram_summary(X, y, bounds=dataset.bounds)

    assert summary.n_pairs > 0
    assert np.isfinite(summary.nugget)
    assert np.isfinite(summary.sill)
    assert np.isfinite(summary.range)
    assert all(np.isfinite(value) for value in summary.length_scale)


def test_kernel_candidates_are_valid_sklearn_kernels() -> None:
    dataset = get_dataset("quadratic_bowl_2d")
    X = dataset.initial_design(n=6, random_state=22)
    y = dataset.evaluate(X)
    summary = estimate_variogram_summary(X, y, bounds=dataset.bounds)

    candidates = make_kernel_candidates(summary)

    assert {family for family, _ in candidates} == {
        "matern_0.5",
        "matern_1.5",
        "matern_2.5",
        "rbf",
        "rational_quadratic",
    }
    assert all(hasattr(kernel, "get_params") for _, kernel in candidates)


def test_adaptive_kernel_selection_is_reproducible() -> None:
    dataset = get_dataset("quadratic_bowl_2d")
    X = dataset.initial_design(n=7, random_state=23)
    y = dataset.evaluate(X)
    config = GprConfig(n_restarts_optimizer=0, random_state=23)
    prior_config = KernelPriorConfig(random_state=23)

    _, first = fit_prior_optimized_gpr(X, y, config, prior_config, dataset.bounds)
    _, second = fit_prior_optimized_gpr(X, y, config, prior_config, dataset.bounds)

    assert first.selected_family == second.selected_family
    assert first.selected_kernel_repr == second.selected_kernel_repr
    assert np.isfinite(first.best_score)
    assert first.score_margin >= 0.0 or np.isnan(first.score_margin)


def test_fixed_kernel_bypasses_adaptive_selection() -> None:
    dataset = get_dataset("quadratic_bowl_2d")
    X = dataset.initial_design(n=5, random_state=24)
    y = dataset.evaluate(X)
    optimizer = KrispUOptimizer(
        dataset.space(),
        gpr_config=GprConfig(
            kernel=RBF(length_scale=1.0),
            n_restarts_optimizer=0,
        ),
    )

    optimizer.fit(X, y)

    assert optimizer.kernel_prior_result_ is None
    assert optimizer.model_selection_history_ == []


def test_recommend_next_returns_kernel_metadata() -> None:
    dataset = get_dataset("quadratic_bowl_2d")
    X = dataset.initial_design(n=6, random_state=25)
    y = dataset.evaluate(X)
    space = ContinuousSpace(dataset.bounds, names=dataset.labels)

    result = recommend_next(
        X,
        y,
        space=space,
        n_recommendations=1,
        n_candidates=24,
        random_state=25,
        gpr_config=GprConfig(n_restarts_optimizer=0, random_state=25),
    )

    assert result.kernel_prior_result is not None
    assert result.selected_kernel_family is not None
    assert result.model_metadata() is not None


def test_tell_reoptimizes_kernel_prior() -> None:
    dataset = get_dataset("quadratic_bowl_2d")
    X = dataset.initial_design(n=5, random_state=26)
    y = dataset.evaluate(X)
    optimizer = KrispUOptimizer(
        dataset.space(),
        n_candidates=24,
        random_state=26,
        optimize_continuous_acquisition=False,
        gpr_config=GprConfig(n_restarts_optimizer=0, random_state=26),
    )
    optimizer.fit(X, y)
    first_history_length = len(optimizer.model_selection_history_)
    acquisition = optimizer.ask()
    next_y = float(dataset.evaluate(acquisition.x_next.reshape(1, -1))[0])

    optimizer.tell(acquisition.x_next, next_y)

    assert optimizer.kernel_prior_result_ is not None
    assert len(optimizer.model_selection_history_) == first_history_length + 1
