"""Error, kernel-selection, and cross-method summary figures."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .styles import METHOD_COLORS, ORANGE, PURPLE, save_figure, style_axis
from .toy_figures import ToyRun, build_toy_run

METHODS = ("KRISP-U", "posterior std", "Maximin", "Random")
METHOD_TO_RUNNER = {
    "KRISP-U": "support_adjusted_krispu",
    "posterior std": "posterior_std",
    "Maximin": "maximin",
    "Random": "random",
}


def build_comparison_runs(
    seed: int,
    *,
    final_budget: int = 20,
    grid_size: int = 48,
    candidate_count: int = 280,
    principal_runs: dict[str, ToyRun] | None = None,
) -> dict[str, dict[str, ToyRun]]:
    """Run the same toy fields with each acquisition method."""

    result: dict[str, dict[str, ToyRun]] = {}
    for field_index, field_name in enumerate(("smooth", "rough_multiscale")):
        field_runs: dict[str, ToyRun] = {}
        for method_index, method_name in enumerate(METHODS):
            runner = METHOD_TO_RUNNER[method_name]
            if method_name == "KRISP-U" and principal_runs is not None:
                field_runs[method_name] = principal_runs[field_name]
            else:
                field_runs[method_name] = build_toy_run(
                    field_name,
                    seed + field_index * 100 + method_index * 7,
                    final_budget=final_budget,
                    grid_size=grid_size,
                    candidate_count=candidate_count,
                    method=runner,
                )
        result[field_name] = field_runs
    return result


def generate_summary_figures(
    output_dir: Path,
    comparison_runs: dict[str, dict[str, ToyRun]],
    *,
    data_root: Path,
    dpi: int = 180,
) -> dict[str, Path]:
    """Write error-v-points, kernel-selection, score, and summary figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    event_rows = _read_kernel_rows(data_root, "events.csv")
    score_rows = _read_kernel_rows(data_root, "candidate_scores.csv")
    return {
        "error_vs_points": save_figure(
            _error_figure(comparison_runs), output_dir / "error_vs_points.png", dpi=dpi
        ),
        "kernel_history": save_figure(
            _kernel_history_figure(event_rows), output_dir / "kernel_selection_history.png", dpi=dpi
        ),
        "kernel_scores": save_figure(
            _kernel_score_figure(score_rows), output_dir / "kernel_scores.png", dpi=dpi
        ),
        "cross_method": save_figure(
            _cross_method_figure(comparison_runs), output_dir / "cross_method_summary.png", dpi=dpi
        ),
    }


def _error_figure(comparison_runs: dict[str, dict[str, ToyRun]]) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for axis, field_name in zip(axes, ("smooth", "rough_multiscale"), strict=True):
        for method in METHODS:
            states = comparison_runs[field_name][method].states
            counts = [state.sample_count for state in states]
            errors = [state.metrics.nrmse for state in states]
            axis.plot(
                counts,
                errors,
                marker="o",
                markersize=3.5,
                linewidth=2.0,
                color=METHOD_COLORS[method],
                label=method,
            )
        axis.set(
            title="smooth field" if field_name == "smooth" else "rough / multiscale field",
            xlabel="number of measurements",
            ylabel="NRMSE",
        )
        axis.set_ylim(bottom=0.0)
        style_axis(axis)
    axes[1].legend(fontsize=8, loc="best")
    figure.suptitle("Error versus number of points", fontsize=15, fontweight="bold")
    return figure


def _cross_method_figure(comparison_runs: dict[str, dict[str, ToyRun]]) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    fields = ("smooth", "rough_multiscale")
    labels = ("smooth", "rough / multiscale")
    x = np.arange(len(METHODS))
    width = 0.36
    for axis, field_name, label in zip(axes, fields, labels, strict=True):
        finals = []
        aucs = []
        for method in METHODS:
            states = comparison_runs[field_name][method].states
            counts = np.asarray([state.sample_count for state in states], dtype=float)
            errors = np.asarray([state.metrics.nrmse for state in states], dtype=float)
            finals.append(float(errors[-1]))
            aucs.append(float(np.trapz(errors, counts) / (counts[-1] - counts[0])))
        axis.bar(
            x - width / 2,
            finals,
            width,
            color=[METHOD_COLORS[method] for method in METHODS],
            label="final NRMSE",
        )
        axis.bar(
            x + width / 2,
            aucs,
            width,
            color=[METHOD_COLORS[method] for method in METHODS],
            alpha=0.35,
            edgecolor=[METHOD_COLORS[method] for method in METHODS],
            linewidth=1.2,
            label="curve AUC / budget",
        )
        axis.set(title=label, ylabel="NRMSE", xticks=x, xticklabels=METHODS)
        axis.tick_params(axis="x", rotation=35)
        axis.set_ylim(bottom=0.0)
        style_axis(axis)
    axes[0].legend(fontsize=8)
    figure.suptitle("Cross-method reconstruction summary", fontsize=15, fontweight="bold")
    return figure


def _kernel_history_figure(rows: list[dict[str, str]]) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    if not rows:
        for axis in axes:
            axis.text(0.5, 0.5, "No kernel-selection table found", ha="center", va="center")
            axis.axis("off")
        return figure
    fields = sorted({row.get("field", "field") for row in rows})[:5]
    for field in fields:
        values = sorted(
            [row for row in rows if row.get("field") == field],
            key=lambda row: int(row.get("sample_count", 0)),
        )
        axes[0].step(
            [int(row["sample_count"]) for row in values],
            [row.get("selected_family", row.get("selected_kernel_id", "unknown")) for row in values],
            where="post",
            label=field,
        )
    axes[0].set(title="Selected kernel family", xlabel="number of measurements", ylabel="family")
    axes[0].tick_params(axis="y", labelsize=7)
    axes[0].legend(fontsize=7, loc="best")
    axes[0].grid(alpha=0.25)

    transitions = Counter(
        row.get("selected_family", row.get("selected_kernel_id", "unknown")) for row in rows
    )
    names = list(transitions)
    axes[1].bar(names, [transitions[name] for name in names], color=PURPLE)
    axes[1].set(title="Selection events by family", ylabel="event count")
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    style_axis(axes[1])
    figure.suptitle("Kernel-selection history", fontsize=15, fontweight="bold")
    return figure


def _kernel_score_figure(rows: list[dict[str, str]]) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    if not rows:
        for axis in axes:
            axis.text(0.5, 0.5, "No candidate-score table found", ha="center", va="center")
            axis.axis("off")
        return figure
    field = min({row.get("field", "field") for row in rows})
    subset = [row for row in rows if row.get("field") == field and row.get("trial", "0") == "0"]
    for kernel_id in sorted({row.get("candidate_kernel_id", "unknown") for row in subset}):
        values = sorted(
            [row for row in subset if row.get("candidate_kernel_id") == kernel_id],
            key=lambda row: int(row.get("sample_count", 0)),
        )
        if not values:
            continue
        axes[0].plot(
            [int(row["sample_count"]) for row in values],
            [float(row["selection_score"]) for row in values],
            ".-",
            linewidth=1.4,
            label=kernel_id,
        )
    axes[0].set(title=f"Buffered scores | {field}", xlabel="number of measurements", ylabel="lower is better")
    axes[0].legend(fontsize=7, ncol=2)
    style_axis(axes[0])

    best_by_count: dict[int, float] = {}
    for row in subset:
        count = int(row["sample_count"])
        score = float(row["selection_score"])
        best_by_count[count] = min(score, best_by_count.get(count, np.inf))
    counts = sorted(best_by_count)
    axes[1].plot(counts, [best_by_count[count] for count in counts], "o-", color=ORANGE, linewidth=2)
    axes[1].set(title="Best validated candidate", xlabel="number of measurements", ylabel="score")
    style_axis(axes[1])
    figure.suptitle("Kernel candidate scores", fontsize=15, fontweight="bold")
    return figure


def _read_kernel_rows(data_root: Path, filename: str) -> list[dict[str, str]]:
    candidates = (
        data_root / "kernel_recovery" / "kernel" / filename,
        data_root / "canonical_2d" / "kernel" / filename,
        data_root / "noise_robustness" / "kernel" / filename,
    )
    for path in candidates:
        if path.exists():
            with path.open(encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
    return []


__all__ = ["build_comparison_runs", "generate_summary_figures"]
