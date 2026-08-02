"""Headless Matplotlib plots for the v0.3.0 performance audit."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_field_audit(state: Any, path: Path, save_pdf: bool = False) -> None:
    x, y, shape = _grid(state.evaluation_points)
    true = state.true_field.reshape(shape)
    predicted = state.predicted_field.reshape(shape)
    error = state.metrics.absolute_error.reshape(shape)
    if state.krispu_uncertainty is None:
        raise ValueError("The field audit requires KRISP-U uncertainty.")
    fields = [true, predicted, state.krispu_uncertainty.reshape(shape), error]
    labels = [
        "True field",
        "Reconstructed field",
        "KRISP-U uncertainty",
        "Absolute reconstruction error",
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    related_limits = (float(np.min(true)), float(np.max(true)))
    error_limit = max(float(np.max(error)), np.finfo(float).eps)
    axes = np.asarray(axes).reshape(-1)
    for axis, values, label in zip(axes.flat, fields, labels, strict=True):
        if label.startswith(("True", "Reconstructed")):
            limits = related_limits
        elif label == "KRISP-U uncertainty":
            limits = (0.0, max(float(np.max(values)), np.finfo(float).eps))
        else:
            limits = (0.0, error_limit)
        image = axis.pcolormesh(
            x, y, values, shading="auto", cmap="viridis", vmin=limits[0], vmax=limits[1]
        )
        axis.scatter(
            state.observed_X[:, 0],
            state.observed_X[:, 1],
            s=18,
            c="white",
            edgecolors="black",
            linewidths=0.5,
        )
        if state.recommended_point is not None:
            axis.scatter(*state.recommended_point, marker="*", s=80, c="red", edgecolors="black")
        axis.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal", title=label)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.suptitle(f"{state.field} | {state.method} | n={state.sample_count}")
    _save(figure, path, save_pdf)


def plot_uncertainty_components(state: Any, path: Path, save_pdf: bool = False) -> None:
    if state.loo_field_sensitivity is None:
        return
    x, y, shape = _grid(state.evaluation_points)
    values = [
        state.loo_field_sensitivity,
        state.kernel_support_deficit,
        state.krispu_uncertainty,
        state.metrics.absolute_error,
    ]
    labels = [
        "LOO field sensitivity",
        "Kernel support deficit",
        "KRISP-U uncertainty",
        "absolute error",
    ]
    figure, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for axis, value, label in zip(axes.flat, values, labels, strict=True):
        image = axis.pcolormesh(x, y, value.reshape(shape), shading="auto", cmap="magma", vmin=0.0)
        axis.scatter(
            state.observed_X[:, 0],
            state.observed_X[:, 1],
            s=14,
            c="cyan",
            edgecolors="black",
            linewidths=0.4,
        )
        if state.recommended_point is not None:
            axis.scatter(
                *state.recommended_point,
                marker="*",
                s=90,
                c="red",
                edgecolors="black",
            )
        axis.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal", title=label)
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.suptitle(f"Uncertainty components | {state.field} | n={state.sample_count}")
    _save(figure, path, save_pdf)


def plot_learning_curves(
    records: list[dict[str, Any]], directory: Path, paired: bool = False
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("nrmse", "NRMSE"),
        ("r2", "R²"),
        ("p95_absolute_error", "p95 absolute error"),
        ("max_absolute_error", "maximum absolute error"),
    ]
    fields = sorted({str(row["field"]) for row in records})
    for field in fields:
        figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        field_rows = [row for row in records if row["field"] == field]
        for axis, (key, title) in zip(axes.flat, metrics, strict=True):
            for method in sorted({str(row["method"]) for row in field_rows}):
                rows = [row for row in field_rows if row["method"] == method]
                grouped: dict[int, list[float]] = {}
                for row in rows:
                    grouped.setdefault(int(row["sample_count"]), []).append(float(row[key]))
                counts = np.array(sorted(grouped))
                values = np.array([grouped[count] for count in counts], dtype=float)
                if paired and values.shape[1] > 1:
                    axis.plot(counts, np.median(values, axis=1), label=method)
                    axis.fill_between(
                        counts,
                        np.percentile(values, 25, axis=1),
                        np.percentile(values, 75, axis=1),
                        alpha=0.18,
                    )
                else:
                    axis.plot(counts, values[:, 0], label=method)
            axis.set(title=title, xlabel="number of measurements", ylabel=title)
            axis.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=8)
        figure.suptitle(f"Learning curves | {field}")
        _save(figure, directory / f"{field}_learning_curves.png", paired)


def plot_sampling_paths(
    field: Any, final_states: dict[str, Any], path: Path, save_pdf: bool = False
) -> None:
    n = len(final_states)
    figure, axes = plt.subplots(2, math.ceil(n / 2), figsize=(13, 7), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    grid_points = _regular_grid(field.domain, 50)
    values = field.evaluate(grid_points)
    _, _, shape = _grid(grid_points)
    levels = np.linspace(float(np.min(values)), float(np.max(values)), 16)
    for axis, (method, state) in zip(axes, final_states.items(), strict=False):
        X, Y = grid_points[:, 0].reshape(shape), grid_points[:, 1].reshape(shape)
        axis.contourf(X, Y, values.reshape(shape), levels=levels, cmap="viridis")
        initial_count = state.initial_sample_count
        axis.scatter(
            state.observed_X[:initial_count, 0],
            state.observed_X[:initial_count, 1],
            c="white",
            edgecolors="black",
            s=25,
            label="initial",
        )
        if len(state.observed_X) > initial_count:
            order = np.arange(len(state.observed_X) - initial_count)
            points = state.observed_X[initial_count:]
            scatter = axis.scatter(
                points[:, 0], points[:, 1], c=order, cmap="plasma", s=28, label="adaptive"
            )
            figure.colorbar(scatter, ax=axis, shrink=0.75, label="adaptive order")
        axis.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal", title=method)
    for axis in axes[n:]:
        axis.axis("off")
    figure.suptitle(f"Sampling paths | {field.name}")
    _save(figure, path, save_pdf)


def plot_uncertainty_error(state: Any, path: Path, save_pdf: bool = False) -> None:
    pairs = [
        ("posterior_std", state.posterior_std),
        ("loo_field_sensitivity", state.loo_field_sensitivity),
        ("kernel_support_deficit", state.kernel_support_deficit),
        ("krispu_uncertainty", state.krispu_uncertainty),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for axis, (label, uncertainty) in zip(axes.flat, pairs, strict=True):
        if uncertainty is None:
            axis.axis("off")
            continue
        axis.scatter(uncertainty, state.metrics.absolute_error, s=5, alpha=0.35)
        axis.set(
            xlabel=label,
            ylabel="absolute reconstruction error",
            title=_correlations(uncertainty, state.metrics.absolute_error),
        )
        axis.grid(alpha=0.2)
    figure.suptitle(f"Uncertainty versus actual error | {state.field} | n={state.sample_count}")
    _save(figure, path, save_pdf)


def plot_error_concentration(state: Any, path: Path, save_pdf: bool = False) -> None:
    candidates = [
        ("loo_field_sensitivity", state.loo_field_sensitivity),
        ("krispu_uncertainty", state.krispu_uncertainty),
        ("posterior_std", state.posterior_std),
    ]
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    total = float(np.sum(state.metrics.squared_error))
    for label, uncertainty in candidates:
        if uncertainty is None:
            continue
        order = np.argsort(-uncertainty)
        axis.plot(
            np.arange(1, len(order) + 1) / len(order),
            np.cumsum(state.metrics.squared_error[order]) / total,
            label=label,
        )
    fraction = np.arange(1, len(state.metrics.squared_error) + 1) / len(state.metrics.squared_error)
    axis.plot(fraction, fraction, "k--", label="random ordering expectation")
    axis.set(
        xlabel="fraction of domain inspected",
        ylabel="fraction of squared error captured",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    axis.grid(alpha=0.25)
    _save(figure, path, save_pdf)


def plot_component_evolution(
    records: list[dict[str, Any]], path: Path, save_pdf: bool = False
) -> None:
    rows = [row for row in records if row["method"] == "support_adjusted_krispu"]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for key in (
        "mean_posterior_std",
        "mean_loo_field_sensitivity",
        "mean_kernel_support_deficit",
        "mean_krispu_uncertainty",
    ):
        values = [
            (int(row["sample_count"]), float(row[key]))
            for row in rows
            if row[key] not in (None, "")
        ]
        if values:
            values.sort()
            axis.plot(*zip(*values), marker="o", label=key)
    axis.set(
        xlabel="number of measurements",
        ylabel="mean uncertainty",
        title="KRISP-U component evolution",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    _save(figure, path, save_pdf)


def plot_boundary_diagnostics(
    records: list[dict[str, Any]], path: Path, save_pdf: bool = False
) -> None:
    """Plot boundary, hull, and nearest-observation selection diagnostics."""

    adaptive = [row for row in records if row["recommended_x"] not in (None, "")]
    counts = np.asarray([int(row["sample_count"]) for row in adaptive])
    boundary = np.asarray([bool(row["near_domain_boundary"]) for row in adaptive], dtype=float)
    hull = np.asarray([bool(row["on_current_sample_hull"]) for row in adaptive], dtype=float)
    distances = np.asarray(
        [float(row["nearest_normalized_distance"]) for row in adaptive], dtype=float
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    if len(counts):
        axes[0].plot(counts, np.cumsum(boundary) / np.arange(1, len(boundary) + 1), marker="o")
        axes[1].plot(counts, np.cumsum(hull) / np.arange(1, len(hull) + 1), marker="o")
        axes[2].plot(counts, distances, marker="o")
    axes[0].set(title="Selections near domain boundary", ylabel="fraction")
    axes[1].set(title="Selections becoming hull vertices", ylabel="fraction")
    axes[2].set(title="Nearest-observation distance", ylabel="normalized distance")
    for axis in axes:
        axis.set_xlabel("sample count")
        axis.grid(alpha=0.25)
    _save(figure, path, save_pdf)


def plot_dominant_loo_diagnostics(states: list[Any], path: Path, save_pdf: bool = False) -> None:
    """Show LOO-spread maps and the fold driving selected iterations."""

    selected = [state for state in states if state.recommended_point is not None]
    if not selected:
        return
    selected = selected[:: max(1, len(selected) // 4)][:4]
    figure, axes = plt.subplots(
        1, len(selected), figsize=(4 * len(selected), 4), constrained_layout=True
    )
    axes = np.asarray(axes).reshape(-1)
    for axis, state in zip(axes, selected, strict=True):
        x, y, shape = _grid(state.evaluation_points)
        values = state.loo_field_sensitivity.reshape(shape)
        image = axis.pcolormesh(x, y, values, shading="auto", cmap="magma")
        eligible = state.observed_loo_eligible
        axis.scatter(
            state.observed_X[eligible, 0],
            state.observed_X[eligible, 1],
            c="white",
            edgecolors="black",
            s=24,
            label="LOO eligible",
        )
        if np.any(~eligible):
            axis.scatter(
                state.observed_X[~eligible, 0],
                state.observed_X[~eligible, 1],
                c="cyan",
                marker="s",
                edgecolors="black",
                s=28,
                label="protected anchor",
            )
        axis.scatter(
            *state.recommended_point,
            c="red",
            marker="*",
            s=100,
            edgecolors="black",
            label="next selected",
        )
        dominant = state.dominant_loo_observation_index
        title = f"n={state.sample_count}; dominant fold={dominant}"
        axis.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal", title=title)
        axis.legend(fontsize=7, loc="best")
        figure.colorbar(image, ax=axis, shrink=0.8, label="LOO field sensitivity")
    figure.suptitle(f"Dominant LOO fold diagnostics | {selected[0].field}")
    _save(figure, path, save_pdf)


def plot_paired_differences(
    rows: list[dict[str, Any]], directory: Path, save_pdf: bool = False
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for baseline in sorted({str(row["baseline"]) for row in rows}):
        values = np.array(
            [float(row["delta_nrmse"]) for row in rows if row["baseline"] == baseline]
        )
        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.scatter(np.arange(len(values)), values)
        if len(values):
            median = float(np.median(values))
            axis.axhline(median, color="tab:red", linestyle="--", label=f"median={median:.3g}")
            axis.text(
                0.02,
                0.95,
                f"win percentage: {100 * np.mean(values < 0):.1f}%",
                transform=axis.transAxes,
                va="top",
            )
        axis.set(
            title=f"Paired NRMSE difference: KRISP-U - {baseline}",
            xlabel="paired trial",
            ylabel="ΔNRMSE",
        )
        axis.legend()
        _save(figure, directory / f"delta_nrmse_vs_{baseline}.png", save_pdf)


def _grid(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    axis = np.unique(points[:, 0])
    other = np.unique(points[:, 1])
    if len(axis) * len(other) != len(points):
        raise ValueError("Plotting requires a complete rectangular evaluation grid.")
    return axis, other, (len(other), len(axis))


def _regular_grid(domain: Any, size: int) -> np.ndarray:
    axes = [np.linspace(lo, hi, size) for lo, hi in domain.bounds]
    mesh = np.meshgrid(*axes, indexing="xy")
    return np.column_stack([item.ravel() for item in mesh])


def _correlations(x: np.ndarray, y: np.ndarray) -> str:
    from scipy.stats import pearsonr, spearmanr

    return f"Pearson={pearsonr(x, y).statistic:.2f}, Spearman={spearmanr(x, y).statistic:.2f}"


def _save(figure: Any, path: Path, save_pdf: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    if save_pdf:
        figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
