"""The compact, fixed-output summary figure suite."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from evaluation.figures.sequential import panel_figure


def write_summary_figures(
    field_states: Mapping[str, Mapping[str, list[Any]]],
    records: list[dict[str, Any]],
    output: Path,
    *,
    dpi: int = 150,
) -> None:
    """Write exactly four figures per two-dimensional field and four global figures."""

    field_root = output / "figures" / "fields"
    global_root = output / "figures" / "global"
    for field_name, method_states in field_states.items():
        states = _principal_states(method_states)
        if not states or states[0].evaluation_points.shape[1] != 2:
            continue
        field_output = field_root / field_name
        field_output.mkdir(parents=True, exist_ok=True)
        _write_process_gif(states, field_output / "process.gif")
        _write_checkpoints(states, field_output / "checkpoints.png", dpi)
        _write_learning_curve(
            [row for row in records if row["field"] == field_name],
            field_output / "learning_curve.png",
            dpi,
        )
        _write_kernel_history(states, field_output / "kernel_history.png", dpi)
    _write_global_figures(records, global_root, dpi)


def _principal_states(method_states: Mapping[str, list[Any]]) -> list[Any]:
    for name in ("krispu_adaptive", "support_adjusted_krispu", "krispu_fixed_matern32"):
        if name in method_states:
            return list(method_states[name])
    return list(next(iter(method_states.values()), []))


def _write_process_gif(states: list[Any], path: Path) -> None:
    frames: list[Image.Image] = []
    scales = _run_scales(states)
    for state in states:
        figure = panel_figure(state, scales, annotate_point_order=False)
        figure.canvas.draw()
        frames.append(
            Image.fromarray(np.asarray(figure.canvas.buffer_rgba()).copy()).convert("RGB")
        )
        plt.close(figure)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=500, loop=0)


def _write_checkpoints(states: list[Any], path: Path, dpi: int) -> None:
    indexes = sorted({0, len(states) // 2, len(states) - 1})
    figure, axes = plt.subplots(len(indexes), 4, figsize=(13, 3.5 * len(indexes)), squeeze=False)
    scales = _run_scales(states)
    for row_index, state_index in enumerate(indexes):
        state = states[state_index]
        x, y = _grid_axes(state.evaluation_points)
        uncertainty = _state_uncertainty(state)
        values = (
            state.true_field,
            state.predicted_field,
            uncertainty,
            state.metrics.absolute_error,
        )
        titles = ("Truth", "Reconstruction", "KRISP-U uncertainty", "Absolute error")
        limits = (scales["true"], scales["true"], scales["uncertainty"], scales["error"])
        for column, (value, title, limit) in enumerate(zip(values, titles, limits, strict=True)):
            image = axes[row_index, column].pcolormesh(
                x,
                y,
                np.asarray(value).reshape(len(y), len(x)),
                shading="auto",
                cmap="viridis" if column < 2 else "magma",
                vmin=limit[0],
                vmax=limit[1],
            )
            axes[row_index, column].scatter(
                state.observed_X[:, 0], state.observed_X[:, 1], c="white", edgecolors="black", s=18
            )
            axes[row_index, column].set(title=title, aspect="equal")
            figure.colorbar(image, ax=axes[row_index, column], shrink=0.75)
        result = state.kernel_selection_result
        reasons = (
            "none"
            if result is None
            else "; ".join(result.reselection_event.reselection_reasons) or "none"
        )
        switch = (
            "none"
            if result is None
            else (result.previous_kernel_id or "initial") + " -> " + result.selected_kernel_id
        )
        axes[row_index, 0].set_ylabel(
            f"n={state.sample_count}\nfamily={state.selected_kernel_id}\n"
            f"ARD={_format_scales(state.current_length_scales)}\nreselection={reasons}\nswitch={switch}",
            fontsize=8,
        )
    figure.suptitle(f"{states[0].field} checkpoints")
    _save(figure, path, dpi)


def _write_learning_curve(records: list[dict[str, Any]], path: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for method in sorted({str(row["method"]) for row in records}):
        trial_curves = _trial_curves(records, method, "nrmse")
        if not trial_curves:
            continue
        counts = np.asarray(
            sorted({count for curve in trial_curves for count in curve}), dtype=float
        )
        matrix = np.asarray(
            [[curve.get(int(count), np.nan) for count in counts] for curve in trial_curves]
        )
        median = np.nanmedian(matrix, axis=0)
        low, high = np.nanpercentile(matrix, [25, 75], axis=0)
        axis.plot(counts, median, label=method)
        axis.fill_between(counts, low, high, alpha=0.15)
    events = [row for row in records if _as_bool(row.get("reselection_triggered"))]
    locations = [int(row["sample_count"]) for row in events]
    if locations and len(set(locations)) <= max(3, len(locations) // 3):
        for location in sorted(set(locations)):
            axis.axvline(location, color="black", alpha=0.18, linewidth=0.7)
    if locations and len(set(locations)) > max(3, len(locations) // 3):
        frequency: dict[int, int] = {}
        for location in locations:
            frequency[location] = frequency.get(location, 0) + 1
        strip = axis.inset_axes([0.0, -0.20, 1.0, 0.12], sharex=axis)
        strip.bar(
            list(frequency),
            list(frequency.values()),
            width=0.35,
            color="black",
            alpha=0.75,
            label="reselection frequency",
        )
        strip.set_ylabel("events", fontsize=7)
        strip.tick_params(axis="both", labelsize=7)
        strip.grid(alpha=0.15)
    switches = [int(row["sample_count"]) for row in records if _as_bool(row.get("switch_accepted"))]
    if switches:
        axis.scatter(
            switches,
            np.full(len(switches), 0.04),
            marker="o",
            facecolors="black",
            edgecolors="black",
            transform=axis.get_xaxis_transform(),
            label="accepted switch",
        )
    axis.set(xlabel="sample count", ylabel="NRMSE", title="Learning curve")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    _save(figure, path, dpi)


def _write_kernel_history(states: list[Any], path: Path, dpi: int) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    counts = [state.sample_count for state in states]
    axes[0].step(counts, [state.selected_kernel_id for state in states], where="post")
    axes[0].set_ylabel("family")
    scale_rows = [state.current_length_scales for state in states]
    for dimension in range(max((len(row) for row in scale_rows), default=0)):
        values = [row[dimension] if len(row) > dimension else np.nan for row in scale_rows]
        axes[1].plot(counts, values, marker=".", label=f"x{dimension + 1}")
    for state in states:
        result = state.kernel_selection_result
        if result is None:
            continue
        for dimension, (lower, upper) in enumerate(
            zip(
                result.optimization_event.length_scale_minimums,
                result.optimization_event.length_scale_maximums,
                strict=False,
            )
        ):
            if state is states[0]:
                axes[1].axhline(lower, color=f"C{dimension}", linestyle=":", alpha=0.5, label=f"x{dimension + 1} min")
                axes[1].axhline(upper, color=f"C{dimension}", linestyle="--", alpha=0.5, label=f"x{dimension + 1} max")
    axes[1].set_ylabel("ARD length scale")
    current = [
        state.selection_score if state.selection_score is not None else np.nan for state in states
    ]
    challenger = [_challenger_score(state) for state in states]
    axes[2].plot(counts, current, label="current family")
    axes[2].plot(counts, challenger, label="best challenger")
    axes[2].set_ylabel("buffered score")
    axes[2].set_xlabel("sample count")
    marker_seen: set[str] = set()
    reason_markers = {
        "bound-contact trigger": ("D", "tab:orange"),
        "score-degradation trigger": ("v", "tab:purple"),
        "maximum-interval trigger": ("s", "tab:green"),
        "fit-failure trigger": ("X", "tab:red"),
    }
    for axis in axes:
        axis.grid(alpha=0.25)
        for state in states:
            result = state.kernel_selection_result
            if result is None:
                continue
            if result.reselection_event.reselection_triggered:
                retained = not (
                    result.switch_accepted
                    and result.previous_kernel_id not in (None, result.selected_kernel_id)
                )
                label = "reselection evaluated, retained" if retained else "reselection accepted"
                axis.axvline(
                    state.sample_count,
                    color="black" if retained else "crimson",
                    alpha=0.28,
                    linewidth=0.8,
                    label=label if label not in marker_seen else None,
                )
                marker_seen.add(label)
                for reason in result.reselection_event.reselection_reasons:
                    marker = reason_markers.get(reason)
                    if marker is None:
                        continue
                    marker_shape, color = marker
                    if axis is axes[2]:
                        y_value = _challenger_score(state)
                    elif axis is axes[1]:
                        y_value = float(state.current_length_scales[0])
                    else:
                        y_value = 0.5
                    axis.scatter(
                        state.sample_count,
                        y_value,
                        marker=marker_shape,
                        color=color,
                        s=35,
                        zorder=6,
                        label=reason if reason not in marker_seen else None,
                    )
                    marker_seen.add(reason)
            if result.switch_accepted and result.previous_kernel_id not in (
                None,
                result.selected_kernel_id,
            ):
                axis.scatter(
                    state.sample_count,
                    0.5,
                    marker="*",
                    color="red",
                    zorder=5,
                    label="accepted family switch" if "accepted family switch" not in marker_seen else None,
                )
                marker_seen.add("accepted family switch")
    axes[1].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    axes[0].set_title(f"{states[0].field}: global kernel history")
    _save(figure, path, dpi)


def _write_global_figures(records: list[dict[str, Any]], directory: Path, dpi: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_aggregate_curve(records, directory / "aggregate_learning_curve.png", dpi)
    _write_performance_profile(records, directory / "performance_profile.png", dpi)
    _write_kernel_ablation(records, directory / "kernel_ablation.png", dpi)
    _write_robustness_matrix(records, directory / "robustness_matrix.png", dpi)


def _write_aggregate_curve(records: list[dict[str, Any]], path: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for method in sorted({str(row["method"]) for row in records}):
        curves = _trial_curves(records, method, "nrmse")
        if not curves:
            continue
        normalized = []
        grid = np.linspace(0.0, 1.0, 25)
        for curve in curves:
            counts = np.asarray(sorted(curve), dtype=float)
            values = np.asarray([curve[int(count)] for count in counts], dtype=float)
            normalized.append(np.interp(grid, counts / counts[-1], values))
        matrix = np.asarray(normalized)
        median = np.nanmedian(matrix, axis=0)
        low, high = np.nanpercentile(matrix, [25, 75], axis=0)
        axis.plot(grid, median, label=method)
        axis.fill_between(grid, low, high, alpha=0.12)
    axis.set(
        xlabel="normalized sample budget", ylabel="median NRMSE", title="Aggregate learning curve"
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    _save(figure, path, dpi)


def _write_performance_profile(records: list[dict[str, Any]], path: Path, dpi: int) -> None:
    cases = _case_values_by_budget(records)
    methods = sorted({method for values in cases.values() for method in values})
    fractions = []
    for method in methods:
        wins = []
        for values in cases.values():
            best = min(values.values())
            wins.append(values.get(method, np.inf) <= 1.05 * best)
        fractions.append(np.mean(wins) if wins else 0.0)
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.bar(methods, fractions)
    axis.set(ylabel="fraction within factor 1.05 of best", title="Performance profile")
    axis.tick_params(axis="x", rotation=45)
    _save(figure, path, dpi)


def _write_kernel_ablation(records: list[dict[str, Any]], path: Path, dpi: int) -> None:
    methods = ["krispu_fixed_gaussian", "krispu_fixed_matern32", "krispu_manual", "krispu_adaptive"]
    final_values = {method: [] for method in methods}
    auc_values = {method: [] for method in methods}
    for (field, trial, method), rows in _trial_method_rows(records).items():
        if method not in final_values:
            continue
        rows.sort(key=lambda row: int(row["sample_count"]))
        final_values[method].append(float(rows[-1]["nrmse"]))
        counts = np.asarray([int(row["sample_count"]) for row in rows], dtype=float)
        values = np.asarray([float(row["nrmse"]) for row in rows], dtype=float)
        auc_values[method].append(float(np.sum(values[1:] * np.diff(counts))))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(methods, [np.mean(final_values[m]) if final_values[m] else np.nan for m in methods])
    axes[0].set(ylabel="final NRMSE", title="Final error")
    axes[1].bar(methods, [np.mean(auc_values[m]) if auc_values[m] else np.nan for m in methods])
    axes[1].set(ylabel="NRMSE AUC", title="Sampling-curve area")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
    _save(figure, path, dpi)


def _write_robustness_matrix(records: list[dict[str, Any]], path: Path, dpi: int) -> None:
    fields = [
        "smooth_single_scale",
        "white_noise",
        "baseline_drift",
        "baseline_plus_noise",
        "heteroscedastic_noise",
    ]
    matrix = np.full((len(fields), 6), np.nan)
    for row_index, field in enumerate(fields):
        rows = [
            row
            for row in records
            if row["field"] == field
            and row["method"] == "krispu_adaptive"
            and int(row["sample_count"]) == _max_count(records, field, row["trial"], row["method"])
        ]
        if not rows:
            rows = [
                row
                for row in records
                if row["field"] == field
                and int(row["sample_count"]) == _max_count(
                    records, field, row["trial"], row["method"]
                )
            ]
        if rows:
            matrix[row_index, 0] = np.mean([float(row["nrmse"]) for row in rows])
            matrix[row_index, 1] = np.mean(
                [_numeric(row.get("uncertainty_error_rank_correlation")) for row in rows]
            )
            matrix[row_index, 2] = np.mean(
                [_numeric(row.get("near_neighbor_acquisition_rate")) for row in rows]
            )
            matrix[row_index, 3] = np.mean(
                [float(_as_bool(row.get("reselection_triggered")) or 0) for row in rows]
            )
            matrix[row_index, 4] = np.mean(
                [float(_as_bool(row.get("switch_accepted")) or 0) for row in rows]
            )
            matrix[row_index, 5] = np.mean(
                [_numeric(row.get("wall_time_seconds")) for row in rows]
            )
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest")
    axis.set(
        xticks=range(6),
        xticklabels=[
            "error degradation",
            "alignment",
            "near-neighbor",
            "reselection",
            "switch",
            "runtime",
        ],
        yticks=range(len(fields)),
        yticklabels=fields,
        title="Robustness matrix",
    )
    axis.tick_params(axis="x", rotation=35)
    figure.colorbar(image, ax=axis)
    _save(figure, path, dpi)


def _trial_curves(records: list[dict[str, Any]], method: str, key: str) -> list[dict[int, float]]:
    result: list[dict[int, float]] = []
    groups: dict[tuple[str, int], dict[int, float]] = {}
    for row in records:
        if str(row["method"]) == method:
            groups.setdefault((str(row["field"]), int(row["trial"])), {})[
                int(row["sample_count"])
            ] = float(row[key])
    result.extend(groups.values())
    return result


def _final_case_values(records: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, float]]:
    result: dict[tuple[str, int], dict[str, float]] = {}
    for row in records:
        key = (str(row["field"]), int(row["trial"]))
        result.setdefault(key, {})[str(row["method"])] = float(row["nrmse"])
    return result


def _case_values_by_budget(
    records: list[dict[str, Any]],
) -> dict[tuple[str, int, int], dict[str, float]]:
    result: dict[tuple[str, int, int], dict[str, float]] = {}
    for row in records:
        key = (str(row["field"]), int(row["trial"]), int(row["sample_count"]))
        result.setdefault(key, {})[str(row["method"])] = float(row["nrmse"])
    return result


def _trial_method_rows(
    records: list[dict[str, Any]],
) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    result: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in records:
        key = (str(row["field"]), int(row["trial"]), str(row["method"]))
        result.setdefault(key, []).append(row)
    return result


def _max_count(records: list[dict[str, Any]], field: str, trial: Any, method: str) -> int:
    counts = [
        int(row["sample_count"])
        for row in records
        if row["field"] == field and int(row["trial"]) == int(trial) and row["method"] == method
    ]
    return max(counts, default=-1)


def _challenger_score(state: Any) -> float:
    result = state.kernel_selection_result
    if result is None or result.reselection_event.challenger_validation_score is None:
        return np.nan
    return float(result.reselection_event.challenger_validation_score)


def _state_uncertainty(state: Any) -> np.ndarray:
    if state.acquisition_field is not None:
        return np.asarray(state.acquisition_field)
    if state.krispu_uncertainty is not None:
        return np.asarray(state.krispu_uncertainty)
    return np.asarray(state.posterior_std)


def _run_scales(states: list[Any]) -> dict[str, tuple[float, float]]:
    true = np.concatenate([np.asarray(state.true_field) for state in states])
    prediction = np.concatenate([np.asarray(state.predicted_field) for state in states])
    uncertainty = np.concatenate([_state_uncertainty(state) for state in states])
    error = np.concatenate([np.asarray(state.metrics.absolute_error) for state in states])
    return {
        "true": _limits(np.concatenate((true, prediction))),
        "predicted": _limits(np.concatenate((true, prediction))),
        "uncertainty": (0.0, _positive_max(uncertainty)),
        "error": (0.0, _positive_max(error)),
    }


def _grid_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.unique(points[:, 0]), np.unique(points[:, 1])
    if len(x) * len(y) != len(points):
        raise ValueError("visualization requires a rectangular two-dimensional evaluation grid")
    return x, y


def _limits(values: np.ndarray) -> tuple[float, float]:
    low, high = float(np.min(values)), float(np.max(values))
    if np.isclose(low, high):
        padding = max(abs(low) * 0.05, 1e-6)
        return low - padding, high + padding
    return low, high


def _positive_max(values: np.ndarray) -> float:
    return max(float(np.max(values)), np.finfo(float).eps)


def _format_scales(values: Any) -> str:
    return "[" + ", ".join(f"{float(value):.3g}" for value in values) + "]"


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _numeric(value: Any) -> float:
    if value in (None, ""):
        return np.nan
    return float(value)


def _save(figure: Any, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


__all__ = ["write_summary_figures"]
