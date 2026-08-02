"""Sequential field visualizations and method-comparison figures."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _snapshot_sample_counts(
    final_budget: int,
    initial_sample_count: int,
    snapshot_every: int | None = None,
    snapshot_sample_counts: Iterable[int] | None = None,
) -> tuple[int, ...]:
    """Return valid snapshot counts, always including the initial and final states."""

    if snapshot_sample_counts is not None:
        counts = {int(value) for value in snapshot_sample_counts}
    elif snapshot_every is not None and int(snapshot_every) > 0:
        counts = set(range(initial_sample_count, final_budget + 1, int(snapshot_every)))
    else:
        counts = set()
    counts.update((initial_sample_count, final_budget))
    return tuple(sorted(value for value in counts if initial_sample_count <= value <= final_budget))


def snapshot_sample_counts(
    final_budget: int,
    initial_sample_count: int,
    snapshot_every: int | None = None,
    snapshot_sample_counts: Iterable[int] | None = None,
) -> tuple[int, ...]:
    """Public snapshot scheduling helper."""

    return _snapshot_sample_counts(
        final_budget, initial_sample_count, snapshot_every, snapshot_sample_counts
    )


def save_sequential_visuals(
    states: list[Any],
    output_root: Path,
    *,
    save_gif: bool = True,
    save_snapshots: bool = True,
    snapshot_every: int | None = None,
    snapshot_sample_counts: Iterable[int] | None = None,
    frame_duration_ms: int = 500,
    dpi: int = 150,
    annotate_point_order: bool = True,
    save_point_layout_gif: bool = True,
    save_snapshot_gif: bool = True,
) -> dict[str, list[Path] | Path | None]:
    """Save panel frames, optional GIFs, and optional point-layout GIFs for one run."""

    if not states:
        raise ValueError("states must contain at least one sequential state")
    if frame_duration_ms <= 0 or dpi <= 0:
        raise ValueError("frame_duration_ms and dpi must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    selection_mode = getattr(states[0], "selection_mode", "fixed_generic")
    run_key = f"{selection_mode}_{states[0].method}_{states[0].trial}"
    snapshot_dir = output_root / "snapshots" / states[0].field / run_key
    animation_dir = output_root / "animations" / states[0].field / run_key
    point_dir = output_root / "point_progress" / states[0].field / run_key
    scales = _run_scales(states)
    frames: list[Image.Image] = []
    snapshot_frames: list[Image.Image] = []
    frame_paths: list[Path] = []
    snapshot_counts = set(
        _snapshot_sample_counts(
            states[-1].sample_count,
            states[0].initial_sample_count,
            snapshot_every,
            snapshot_sample_counts,
        )
    )
    for state in states:
        figure = panel_figure(state, scales, annotate_point_order=annotate_point_order)
        if save_gif:
            frames.append(_figure_image(figure))
        if save_snapshots and state.sample_count in snapshot_counts:
            path = snapshot_dir / f"n{state.sample_count:04d}.png"
            _save_figure(figure, path, dpi)
            if save_snapshot_gif:
                snapshot_frames.append(_figure_image(figure))
            frame_paths.append(path)
        plt.close(figure)

    animation_path: Path | None = None
    snapshot_animation_path: Path | None = None
    point_animation_path: Path | None = None
    if save_gif:
        animation_path = animation_dir / "panel.gif"
        _save_gif(frames, animation_path, frame_duration_ms)
    if save_snapshots and save_snapshot_gif:
        snapshot_animation_path = snapshot_dir / "panel_snapshots.gif"
        _save_gif(snapshot_frames, snapshot_animation_path, frame_duration_ms)
    if save_point_layout_gif:
        point_frames = [
            _figure_image(
                point_layout_figure(state, annotate_point_order=annotate_point_order),
                close=True,
            )
            for state in states
        ]
        point_animation_path = point_dir / "points.gif"
        _save_gif(point_frames, point_animation_path, frame_duration_ms)
    return {
        "animation": animation_path,
        "snapshots": frame_paths,
        "snapshot_animation": snapshot_animation_path,
        "point_animation": point_animation_path,
    }


def panel_figure(
    state: Any,
    scales: dict[str, tuple[float, float]] | None = None,
    *,
    annotate_point_order: bool = True,
) -> Any:
    """Build the requested 2x2 true/prediction/uncertainty/error panel."""

    x, y = _grid_axes(state.evaluation_points)
    shape = (len(y), len(x))
    uncertainty, uncertainty_label = _uncertainty(state)
    values = {
        "true": state.true_field.reshape(shape),
        "predicted": state.predicted_field.reshape(shape),
        "uncertainty": uncertainty.reshape(shape),
        "error": state.metrics.absolute_error.reshape(shape),
    }
    scales = _run_scales([state]) if scales is None else scales
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    panels = (
        ("true", "True field", "viridis"),
        ("predicted", "Current reconstruction", "viridis"),
        ("uncertainty", uncertainty_label, "magma"),
        ("error", "Absolute error", "inferno"),
    )
    for axis, (key, label, cmap) in zip(axes.flat, panels, strict=True):
        vmin, vmax = scales[key]
        image = axis.pcolormesh(x, y, values[key], shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        _overlay_points(axis, state, annotate_point_order)
        axis.set(
            xlim=(-1.0, 1.0),
            ylim=(-1.0, 1.0),
            aspect="equal",
            xlabel="x",
            ylabel="y",
            title=label,
        )
        figure.colorbar(image, ax=axis, shrink=0.82)
    scales_text = ", ".join(f"{value:.3g}" for value in state.current_length_scales)
    figure.text(
        0.5,
        0.01,
        f"mode={state.selection_mode} | kernel={state.selected_kernel_id} | "
        f"length scales=[{scales_text}]",
        ha="center",
        fontsize=8,
    )
    figure.suptitle(
        f"{state.method} | {state.field} | trial {state.trial} | "
        f"n={state.sample_count} | NRMSE={state.metrics.nrmse:.4g} | R²={state.metrics.r2:.4g}"
    )
    return figure


def point_layout_figure(state: Any, *, annotate_point_order: bool = True) -> Any:
    """Build a point-only view of the current sequential design."""

    figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
    axis.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), aspect="equal", xlabel="x", ylabel="y")
    axis.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color="black")
    initial_count = state.initial_sample_count
    initial = state.observed_X[:initial_count]
    axis.scatter(initial[:, 0], initial[:, 1], c="white", edgecolors="black", label="initial")
    adaptive = state.observed_X[initial_count:]
    if len(adaptive):
        axis.scatter(
            adaptive[:, 0],
            adaptive[:, 1],
            c=np.arange(len(adaptive)),
            cmap="plasma",
            edgecolors="black",
            label="adaptive",
        )
        if annotate_point_order:
            for order, point in enumerate(adaptive, start=initial_count + 1):
                axis.annotate(
                    str(order), point, xytext=(3, 3), textcoords="offset points", fontsize=8
                )
        axis.scatter(
            adaptive[-1, 0], adaptive[-1, 1], marker="*", s=130, c="red", edgecolors="black"
        )
    axis.set_title(f"{state.method} | {state.field} | trial {state.trial} | n={state.sample_count}")
    axis.legend(fontsize=8, loc="best")
    return figure


def plot_method_comparisons(
    records: list[dict[str, Any]],
    directory: Path,
    *,
    uncertainty_bands: bool = True,
    dpi: int = 150,
) -> list[Path]:
    """Plot six method-comparison diagnostics for every field."""

    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    metrics = (
        ("nrmse", "NRMSE"),
        ("r2", "R²"),
        ("p95_absolute_error", "p95 absolute error"),
        ("max_absolute_error", "maximum absolute error"),
        ("distance_to_nearest_observation", "nearest-observation distance"),
        ("boundary_fraction", "fraction near boundary"),
        ("hull_fraction", "fraction on current hull"),
    )
    fields = sorted({str(row["field"]) for row in records})
    for field in fields:
        figure, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
        rows = [row for row in records if str(row["field"]) == field]
        derived = _derived_diagnostics(rows)
        for axis, (key, title) in zip(axes.flat, metrics, strict=False):
            for method in sorted({str(row["method"]) for row in rows}):
                method_rows = [row for row in derived if row["method"] == method]
                counts, mean, low, high = _summary_curve(method_rows, key)
                if not len(counts):
                    continue
                axis.plot(counts, mean, marker="o", label=method)
                if uncertainty_bands and len(method_rows) > len(counts):
                    axis.fill_between(counts, low, high, alpha=0.16)
            axis.set(title=title, xlabel="sample count", ylabel=title)
            axis.grid(alpha=0.25)
        for axis in axes.flat[len(metrics) :]:
            axis.axis("off")
        axes.flat[0].legend(fontsize=8)
        figure.suptitle(f"Method comparison | {field}")
        path = directory / f"{field}_method_comparison.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def _uncertainty(state: Any) -> tuple[np.ndarray, str]:
    if state.krispu_uncertainty is not None:
        return state.krispu_uncertainty, "KRISP-U uncertainty"
    if state.posterior_std is not None:
        return state.posterior_std, "GP posterior std"
    raise ValueError("A fitted-surrogate uncertainty field is required for visualization.")


def _run_scales(states: list[Any]) -> dict[str, tuple[float, float]]:
    arrays: dict[str, list[np.ndarray]] = {
        key: [] for key in ("true", "predicted", "uncertainty", "error")
    }
    for state in states:
        uncertainty, _ = _uncertainty(state)
        arrays["true"].append(np.asarray(state.true_field))
        arrays["predicted"].append(np.asarray(state.predicted_field))
        arrays["uncertainty"].append(np.asarray(uncertainty))
        arrays["error"].append(np.asarray(state.metrics.absolute_error))
    combined = np.concatenate(arrays["true"] + arrays["predicted"])
    scales = {"true": _limits(combined), "predicted": _limits(combined)}
    scales["uncertainty"] = (0.0, _positive_max(np.concatenate(arrays["uncertainty"])))
    scales["error"] = (0.0, _positive_max(np.concatenate(arrays["error"])))
    return scales


def _limits(values: np.ndarray) -> tuple[float, float]:
    low, high = float(np.min(values)), float(np.max(values))
    if np.isclose(low, high):
        padding = max(abs(low) * 0.05, 1e-6)
        return low - padding, high + padding
    return low, high


def _positive_max(values: np.ndarray) -> float:
    return max(float(np.max(values)), np.finfo(float).eps)


def _overlay_points(axis: Any, state: Any, annotate_point_order: bool) -> None:
    initial_count = state.initial_sample_count
    initial = state.observed_X[:initial_count]
    axis.scatter(initial[:, 0], initial[:, 1], c="white", edgecolors="black", s=26, label="initial")
    adaptive = state.observed_X[initial_count:]
    if len(adaptive):
        axis.scatter(
            adaptive[:, 0], adaptive[:, 1], c="cyan", edgecolors="black", s=28, label="adaptive"
        )
        if annotate_point_order:
            for order, point in enumerate(adaptive, start=initial_count + 1):
                axis.annotate(
                    str(order), point, xytext=(3, 3), textcoords="offset points", fontsize=7
                )
        axis.scatter(
            adaptive[-1, 0], adaptive[-1, 1], marker="*", s=110, c="red", edgecolors="black"
        )


def _grid_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.unique(points[:, 0])
    y = np.unique(points[:, 1])
    if len(x) * len(y) != len(points):
        raise ValueError("Visualization requires a complete rectangular evaluation grid.")
    return x, y


def _figure_image(figure: Any, *, close: bool = False) -> Image.Image:
    figure.canvas.draw()
    image = Image.fromarray(np.asarray(figure.canvas.buffer_rgba()).copy()).convert("RGB")
    if close:
        plt.close(figure)
    return image


def _save_figure(figure: Any, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")


def _save_gif(frames: list[Image.Image], path: Path, duration_ms: int) -> None:
    if not frames:
        raise ValueError("Cannot write a GIF without frames.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def _derived_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    derived: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), int(row["trial"])), []).append(row)
    for (method, trial), trial_rows in grouped.items():
        trial_rows.sort(key=lambda row: int(row["sample_count"]))
        boundary_seen: list[float] = []
        hull_seen: list[float] = []
        for row in trial_rows:
            boundary = _as_bool(row.get("near_domain_boundary"))
            hull = _as_bool(row.get("on_current_sample_hull"))
            if boundary is not None:
                boundary_seen.append(float(boundary))
            if hull is not None:
                hull_seen.append(float(hull))
            copied = dict(row)
            copied["method"] = method
            copied["trial"] = trial
            copied["boundary_fraction"] = np.mean(boundary_seen) if boundary_seen else np.nan
            copied["hull_fraction"] = np.mean(hull_seen) if hull_seen else np.nan
            derived.append(copied)
    return derived


def _summary_curve(
    rows: list[dict[str, Any]], key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        value = row.get(key)
        if value in (None, "") or not np.isfinite(float(value)):
            continue
        grouped.setdefault(int(row["sample_count"]), []).append(float(value))
    if not grouped:
        return np.array([]), np.array([]), np.array([]), np.array([])
    counts = np.asarray(sorted(grouped), dtype=float)
    values = [np.asarray(grouped[int(count)], dtype=float) for count in counts]
    mean = np.asarray([np.mean(value) for value in values])
    low = np.asarray([np.percentile(value, 25) for value in values])
    high = np.asarray([np.percentile(value, 75) for value in values])
    return counts, mean, low, high


def _as_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


__all__ = [
    "panel_figure",
    "plot_method_comparisons",
    "point_layout_figure",
    "save_sequential_visuals",
    "snapshot_sample_counts",
]
