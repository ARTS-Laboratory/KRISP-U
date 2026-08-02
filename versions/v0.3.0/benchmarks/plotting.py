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
    fields = [true, predicted, error, state.posterior_std.reshape(shape)]
    labels = ["True hidden field", "Reconstructed field", "Absolute error", "Posterior std"]
    if state.jackknife_std is not None:
        fields += [state.jackknife_std.reshape(shape), state.combined_std.reshape(shape)]
        labels += ["LOO jackknife uncertainty", "Combined KRISP-U uncertainty"]
    else:
        fields += [np.zeros(shape), np.zeros(shape)]
        labels += [
            "LOO jackknife uncertainty (not applicable)",
            "Combined KRISP-U uncertainty (not applicable)",
        ]
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    related_limits = (float(np.min(true)), float(np.max(true)))
    error_limit = max(float(np.max(error)), np.finfo(float).eps)
    for axis, values, label in zip(axes.flat, fields, labels, strict=True):
        limits = (
            related_limits if label.startswith(("True", "Reconstructed")) else (0.0, error_limit)
        )
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
    if state.combined_std is None:
        return
    x, y, shape = _grid(state.evaluation_points)
    ratio = state.jackknife_std / np.maximum(state.combined_std, 1e-12)
    values = [
        state.posterior_std,
        state.jackknife_std,
        state.calibrated_posterior_std,
        state.combined_std,
        ratio,
        state.metrics.absolute_error,
    ]
    labels = [
        "posterior_std",
        "jackknife_std",
        "calibrated_posterior_std",
        "combined_std",
        "jackknife_std / combined_std",
        "absolute error",
    ]
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
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
        axis.scatter(
            state.observed_X[:5, 0],
            state.observed_X[:5, 1],
            c="white",
            edgecolors="black",
            s=25,
            label="initial",
        )
        if len(state.observed_X) > 5:
            order = np.arange(len(state.observed_X) - 5)
            points = state.observed_X[5:]
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
        ("combined_std", state.combined_std),
        ("posterior_std", state.posterior_std),
        ("jackknife_std", state.jackknife_std),
        ("calibrated_posterior_std", state.calibrated_posterior_std),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
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
        ("combined_std", state.combined_std),
        ("jackknife_std", state.jackknife_std),
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
    rows = [row for row in records if row["method"] == "krispu_combined"]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for key in (
        "mean_posterior_std",
        "mean_jackknife_std",
        "mean_calibrated_posterior_std",
        "mean_combined_std",
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
