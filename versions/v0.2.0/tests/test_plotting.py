from __future__ import annotations

import matplotlib
import numpy as np

from krispu import (
    BenchmarkResult,
    KrispUOptimizer,
    MethodTrace,
    get_dataset,
    run_benchmark,
)
from krispu.models import GprConfig
from krispu.plotting import (
    plot_2d_surface,
    plot_acquisition_map,
    plot_benchmark_comparison,
    plot_best_history,
)

matplotlib.use("Agg")


def test_plotting_smoke() -> None:
    dataset = get_dataset("quadratic_bowl_2d")
    initial_X = dataset.initial_design(n=5, random_state=9)
    optimizer = KrispUOptimizer(
        dataset.space(),
        objective=dataset.objective,
        random_state=9,
        n_candidates=32,
        optimize_continuous_acquisition=False,
        gpr_config=GprConfig(n_restarts_optimizer=0),
    )
    result = optimizer.run(dataset.evaluate, initial_X, n_iterations=1)

    ax_surface = plot_2d_surface(dataset, points_per_axis=20, samples=result.X)
    ax_history = plot_best_history(result)

    assert ax_surface is not None
    assert ax_history is not None


def test_plot_kld_acquisition_map_smoke() -> None:
    dataset = get_dataset("quadratic_bowl_2d")
    initial_X = dataset.initial_design(n=5, random_state=11)
    initial_y = dataset.evaluate(initial_X)
    optimizer = KrispUOptimizer(
        dataset.space(),
        objective=dataset.objective,
        acquisition="kld",
        random_state=11,
        n_candidates=32,
        optimize_continuous_acquisition=False,
        gpr_config=GprConfig(n_restarts_optimizer=0),
    )
    optimizer.fit(initial_X, initial_y)

    ax = plot_acquisition_map(optimizer, points_per_axis=12)

    assert ax is not None


def test_benchmark_plot_smoke() -> None:
    result = run_benchmark(
        "quadratic_bowl_2d",
        methods=("krispu", "random"),
        budget=7,
        n_initial=5,
        n_trials=2,
        random_state=10,
        n_candidates=32,
        optimize_continuous_acquisition=False,
    )

    ax = plot_benchmark_comparison(result)

    assert ax is not None


def test_benchmark_plot_handles_early_stopped_ragged_traces() -> None:
    result = BenchmarkResult(
        dataset_name="synthetic",
        objective="minimize",
        budget=5,
        n_initial=2,
    )
    result.methods["krispu"] = [
        MethodTrace(
            method="krispu",
            seed=1,
            X=np.zeros((3, 2), dtype=float),
            y=np.asarray([3.0, 2.0, 1.0]),
            objective="minimize",
        ),
        MethodTrace(
            method="krispu",
            seed=2,
            X=np.zeros((5, 2), dtype=float),
            y=np.asarray([3.0, 2.5, 2.0, 1.5, 1.0]),
            objective="minimize",
        ),
    ]

    ax = plot_benchmark_comparison(result)

    assert ax is not None
