"""Optional plotting helpers for KRISP-U results."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from krispu.acquisition import acquisition_scores, normalize_acquisition_name
from krispu.benchmarks import BenchmarkResult
from krispu.datasets import ToyDataset
from krispu.optimizer import KrispUOptimizer, OptimizationResult
from krispu.space import DiscreteCandidateSpace


def _pyplot() -> Any:
    import matplotlib.pyplot as plt

    return plt


def _get_axis(ax: Any = None) -> tuple[Any, Any]:
    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots()
    return plt, ax


def _grid_2d(
    bounds: NDArray[np.float64], points_per_axis: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if bounds.shape[0] != 2:
        raise ValueError("This plot requires exactly two dimensions.")
    x_axis = np.linspace(bounds[0, 0], bounds[0, 1], points_per_axis)
    y_axis = np.linspace(bounds[1, 0], bounds[1, 1], points_per_axis)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    return grid_x, grid_y, points


def _labels(labels: tuple[str, ...] | list[str] | None, dimension: int) -> list[str]:
    if labels is None:
        return [f"x{index + 1}" for index in range(dimension)]
    return list(labels)


def plot_1d_prediction(
    optimizer: KrispUOptimizer,
    points: int = 300,
    ax: Any = None,
    label: str = "GPR prediction",
) -> Any:
    """Plot 1D predictive mean and uncertainty band."""

    if optimizer.space.dimension != 1:
        raise ValueError("plot_1d_prediction requires a one-dimensional space.")
    plt, ax = _get_axis(ax)
    bounds = optimizer.space.bounds
    x_grid = np.linspace(bounds[0, 0], bounds[0, 1], points).reshape(-1, 1)
    mean, std = optimizer.predict(x_grid)
    ax.plot(x_grid[:, 0], mean, label=label)
    ax.fill_between(x_grid[:, 0], mean - 1.96 * std, mean + 1.96 * std, alpha=0.25)
    if optimizer.X_train_ is not None and optimizer.y_train_ is not None:
        ax.scatter(
            optimizer.X_train_[:, 0], optimizer.y_train_, color="black", zorder=3
        )
    ax.set_xlabel("x")
    ax.set_ylabel("response")
    ax.legend()
    return ax


def plot_2d_surface(
    dataset: ToyDataset,
    points_per_axis: int = 150,
    samples: NDArray[np.float64] | None = None,
    next_point: NDArray[np.float64] | None = None,
    ax: Any = None,
    cmap: str = "viridis",
) -> Any:
    """Plot a 2D toy-dataset response surface."""

    if dataset.dimension != 2:
        raise ValueError("plot_2d_surface requires a two-dimensional dataset.")
    plt, ax = _get_axis(ax)
    grid_x, grid_y, points = _grid_2d(dataset.bounds, points_per_axis)
    values = dataset.evaluate(points).reshape(grid_x.shape)
    contour = ax.contourf(grid_x, grid_y, values, levels=40, cmap=cmap)
    plt.colorbar(contour, ax=ax, label="response")
    if samples is not None:
        ax.scatter(samples[:, 0], samples[:, 1], color="white", edgecolor="black", s=35)
    if next_point is not None:
        ax.scatter(next_point[0], next_point[1], color="red", marker="*", s=150)
    labels = _labels(dataset.labels, 2)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(dataset.name)
    return ax


def plot_2d_prediction(
    optimizer: KrispUOptimizer,
    points_per_axis: int = 120,
    ax: Any = None,
    cmap: str = "viridis",
) -> Any:
    """Plot a 2D GPR predictive-mean map."""

    if isinstance(optimizer.space, DiscreteCandidateSpace):
        raise ValueError("Use plot_discrete_candidate_acquisition for discrete spaces.")
    if optimizer.space.dimension != 2:
        raise ValueError("plot_2d_prediction requires a two-dimensional space.")
    plt, ax = _get_axis(ax)
    grid_x, grid_y, points = _grid_2d(optimizer.space.bounds, points_per_axis)
    mean, _ = optimizer.predict(points)
    image = ax.contourf(
        grid_x, grid_y, mean.reshape(grid_x.shape), levels=40, cmap=cmap
    )
    plt.colorbar(image, ax=ax, label="prediction")
    _overlay_observations(ax, optimizer)
    ax.set_title("GPR prediction")
    return ax


def plot_2d_uncertainty(
    optimizer: KrispUOptimizer,
    points_per_axis: int = 120,
    ax: Any = None,
    cmap: str = "magma",
) -> Any:
    """Plot a 2D predictive-standard-deviation map."""

    if isinstance(optimizer.space, DiscreteCandidateSpace):
        raise ValueError("Use plot_discrete_candidate_acquisition for discrete spaces.")
    if optimizer.space.dimension != 2:
        raise ValueError("plot_2d_uncertainty requires a two-dimensional space.")
    plt, ax = _get_axis(ax)
    grid_x, grid_y, points = _grid_2d(optimizer.space.bounds, points_per_axis)
    _, std = optimizer.predict(points)
    image = ax.contourf(grid_x, grid_y, std.reshape(grid_x.shape), levels=40, cmap=cmap)
    plt.colorbar(image, ax=ax, label="uncertainty")
    _overlay_observations(ax, optimizer)
    ax.set_title("Predictive uncertainty")
    return ax


def plot_acquisition_map(
    optimizer: KrispUOptimizer,
    acquisition: str | None = None,
    points_per_axis: int = 120,
    ax: Any = None,
    cmap: str = "plasma",
) -> Any:
    """Plot a 2D acquisition-score map."""

    if optimizer.model_ is None or optimizer.y_train_ is None:
        raise ValueError("Call fit() before plotting acquisition scores.")
    if isinstance(optimizer.space, DiscreteCandidateSpace):
        raise ValueError("Use plot_discrete_candidate_acquisition for discrete spaces.")
    if optimizer.space.dimension != 2:
        raise ValueError("plot_acquisition_map requires a two-dimensional space.")
    plt, ax = _get_axis(ax)
    method = normalize_acquisition_name(acquisition or optimizer.acquisition)
    grid_x, grid_y, points = _grid_2d(optimizer.space.bounds, points_per_axis)
    if method == "kld":
        _, _, scores = optimizer._score_candidates(points, method)
    else:
        mean, std = optimizer.predict(points)
        scores = acquisition_scores(
            method,
            mean,
            std,
            optimizer.y_train_,
            objective=optimizer.objective,
            xi=optimizer.xi,
            kappa=optimizer.kappa,
        )
    image = ax.contourf(
        grid_x, grid_y, scores.reshape(grid_x.shape), levels=40, cmap=cmap
    )
    plt.colorbar(image, ax=ax, label="acquisition")
    _overlay_observations(ax, optimizer)
    ax.set_title(f"Acquisition: {method}")
    return ax


def plot_discrete_candidate_acquisition(
    optimizer: KrispUOptimizer,
    acquisition: str | None = None,
    ax: Any = None,
    cmap: str = "plasma",
) -> Any:
    """Plot acquisition scores over a 2D preset candidate pool."""

    if not isinstance(optimizer.space, DiscreteCandidateSpace):
        raise ValueError(
            "plot_discrete_candidate_acquisition requires a discrete space."
        )
    if optimizer.space.dimension != 2:
        raise ValueError("This plot requires two-dimensional candidates.")
    if optimizer.model_ is None or optimizer.y_train_ is None:
        raise ValueError("Call fit() before plotting acquisition scores.")
    plt, ax = _get_axis(ax)
    method = normalize_acquisition_name(acquisition or optimizer.acquisition)
    candidates = optimizer.space.candidates
    if method == "kld":
        _, _, scores = optimizer._score_candidates(candidates, method)
    else:
        mean, std = optimizer.predict(candidates)
        scores = acquisition_scores(
            method,
            mean,
            std,
            optimizer.y_train_,
            objective=optimizer.objective,
            xi=optimizer.xi,
            kappa=optimizer.kappa,
        )
    scatter = ax.scatter(candidates[:, 0], candidates[:, 1], c=scores, cmap=cmap, s=35)
    plt.colorbar(scatter, ax=ax, label="acquisition")
    _overlay_observations(ax, optimizer)
    ax.set_title(f"Discrete acquisition: {method}")
    return ax


def plot_best_history(
    result: OptimizationResult,
    ax: Any = None,
    label: str = "KRISP-U",
) -> Any:
    """Plot a secondary best-observed response trace over iterations."""

    _, ax = _get_axis(ax)
    history = result.best_y_history
    ax.plot(np.arange(1, len(history) + 1), history, marker="o", label=label)
    ax.set_xlabel("evaluation")
    ax.set_ylabel("best observed response")
    ax.legend()
    return ax


def plot_benchmark_comparison(
    result: BenchmarkResult,
    ax: Any = None,
    show_interval: bool = True,
) -> Any:
    """Plot secondary mean best-observed curves for benchmark methods."""

    _, ax = _get_axis(ax)
    for method, traces in result.methods.items():
        histories = [trace.best_y_history for trace in traces]
        max_length = max(len(history) for history in histories)
        x_axis = np.arange(1, max_length + 1)
        mean = np.asarray(
            [
                np.mean(
                    [history[index] for history in histories if len(history) > index]
                )
                for index in range(max_length)
            ],
            dtype=float,
        )
        ax.plot(x_axis, mean, marker="o", label=method)
        if show_interval and len(traces) > 1:
            low = np.asarray(
                [
                    np.quantile(
                        [
                            history[index]
                            for history in histories
                            if len(history) > index
                        ],
                        0.025,
                    )
                    for index in range(max_length)
                ],
                dtype=float,
            )
            high = np.asarray(
                [
                    np.quantile(
                        [
                            history[index]
                            for history in histories
                            if len(history) > index
                        ],
                        0.975,
                    )
                    for index in range(max_length)
                ],
                dtype=float,
            )
            ax.fill_between(x_axis, low, high, alpha=0.15)
    ax.set_xlabel("evaluation")
    ax.set_ylabel("best observed response")
    ax.set_title(result.dataset_name)
    ax.legend()
    return ax


def plot_pairwise_slices(
    dataset: ToyDataset,
    fixed_point: NDArray[np.float64] | None = None,
    dimensions: tuple[int, int] = (0, 1),
    points_per_axis: int = 100,
    ax: Any = None,
    cmap: str = "viridis",
) -> Any:
    """Plot a 2D slice through an n-D toy dataset."""

    if dataset.dimension < 2:
        raise ValueError("Pairwise slices require at least two dimensions.")
    dim_x, dim_y = dimensions
    if dim_x == dim_y or dim_x >= dataset.dimension or dim_y >= dataset.dimension:
        raise ValueError("dimensions must be two distinct valid dimension indices.")
    plt, ax = _get_axis(ax)
    fixed = (
        np.mean(dataset.bounds, axis=1)
        if fixed_point is None
        else np.asarray(fixed_point, dtype=float).reshape(-1)
    )
    if fixed.shape[0] != dataset.dimension:
        raise ValueError("fixed_point must match dataset dimensionality.")
    x_axis = np.linspace(
        dataset.bounds[dim_x, 0], dataset.bounds[dim_x, 1], points_per_axis
    )
    y_axis = np.linspace(
        dataset.bounds[dim_y, 0], dataset.bounds[dim_y, 1], points_per_axis
    )
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    points = np.repeat(fixed.reshape(1, -1), grid_x.size, axis=0)
    points[:, dim_x] = grid_x.ravel()
    points[:, dim_y] = grid_y.ravel()
    values = dataset.evaluate(points).reshape(grid_x.shape)
    contour = ax.contourf(grid_x, grid_y, values, levels=40, cmap=cmap)
    plt.colorbar(contour, ax=ax, label="response")
    labels = _labels(dataset.labels, dataset.dimension)
    ax.set_xlabel(labels[dim_x])
    ax.set_ylabel(labels[dim_y])
    ax.set_title(f"{dataset.name}: {labels[dim_x]} vs {labels[dim_y]}")
    return ax


def _overlay_observations(ax: Any, optimizer: KrispUOptimizer) -> None:
    if optimizer.X_train_ is None:
        return
    if optimizer.X_train_.shape[1] != 2:
        return
    ax.scatter(
        optimizer.X_train_[:, 0],
        optimizer.X_train_[:, 1],
        color="white",
        edgecolor="black",
        s=35,
        zorder=4,
        label="observed",
    )
    ax.legend(loc="best")


__all__ = [
    "plot_1d_prediction",
    "plot_2d_prediction",
    "plot_2d_surface",
    "plot_2d_uncertainty",
    "plot_acquisition_map",
    "plot_benchmark_comparison",
    "plot_best_history",
    "plot_discrete_candidate_acquisition",
    "plot_pairwise_slices",
]
