"""Comparison of common space-filling designs with KRISP-U acquisition."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist, pdist
from scipy.stats import qmc

from .styles import INK, METHOD_COLORS, save_figure, style_axis
from .toy_figures import build_toy_run


def generate_doe_figures(
    output_dir: Path,
    *,
    seed: int = 202603,
    n_points: int = 20,
    krispu_points: np.ndarray | None = None,
    animation_points: int = 36,
    animation_frame_ms: int = 450,
    animation_final_hold_ms: int = 2500,
    dpi: int = 180,
) -> dict[str, Path]:
    """Write the design-layout, coverage, and sampling-progress comparisons."""

    designs = _designs(seed, n_points, krispu_points)
    paths = {
        "layouts": save_figure(
            _layout_figure(designs), output_dir / "doe_methods.png", dpi=dpi
        ),
        "metrics": save_figure(
            _metric_figure(designs), output_dir / "doe_coverage_metrics.png", dpi=dpi
        ),
        "sampling_progress": _write_sampling_gif(
            output_dir / "sampling_progress.gif",
            seed=seed,
            n_points=animation_points,
            frame_duration_ms=animation_frame_ms,
            final_hold_ms=animation_final_hold_ms,
            dpi=dpi,
        ),
    }
    return paths


def _designs(seed: int, n_points: int, krispu_points: np.ndarray | None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    random = rng.uniform(-1.0, 1.0, size=(n_points, 2))
    latin = 2.0 * qmc.LatinHypercube(d=2, seed=seed + 1).random(n_points) - 1.0
    candidates = rng.uniform(-1.0, 1.0, size=(n_points * 80, 2))
    maximin = _greedy_maximin(candidates, n_points, seed + 2)
    if krispu_points is None:
        toy = build_toy_run(
            "smooth",
            seed + 3,
            final_budget=n_points,
            grid_size=32,
            candidate_count=max(160, n_points * 10),
        )
        krispu = np.asarray(toy.final_state.observed_X)
    else:
        krispu = np.asarray(krispu_points, dtype=float)
        if krispu.shape != (n_points, 2):
            raise ValueError(f"krispu_points must have shape {(n_points, 2)}")
    return {
        "Random": random,
        "Latin hypercube": latin,
        "Maximin": maximin,
        "KRISP-U": krispu,
    }


def _greedy_maximin(candidates: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = [int(np.argmin(np.linalg.norm(candidates, axis=1)))]
    while len(selected) < count:
        distances = cdist(candidates, candidates[selected]).min(axis=1)
        distances[selected] = -np.inf
        best = np.flatnonzero(np.isclose(distances, np.max(distances)))
        selected.append(int(rng.choice(best)))
    return candidates[np.asarray(selected)]


def _layout_figure(designs: dict[str, np.ndarray]) -> Any:
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 9.0), constrained_layout=True)
    for axis, (name, points) in zip(axes.ravel(), designs.items(), strict=True):
        color = METHOD_COLORS[name]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=52,
            c=color,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.95,
        )
        for index, point in enumerate(points):
            axis.text(point[0] + 0.025, point[1] + 0.025, str(index + 1), fontsize=7, color=INK)
        metric = _design_metrics(points)
        axis.set(
            title=f"{name}\nminimum spacing = {metric['min_pair']:.2f}",
            xlim=(-1.05, 1.05),
            ylim=(-1.05, 1.05),
            xlabel="x₁",
            ylabel="x₂",
            aspect="equal",
        )
        style_axis(axis, grid=False)
        axis.set_facecolor("white")
    figure.suptitle(
        "Existing DOE approaches: same budget, different inductive bias",
        fontsize=16,
        fontweight="bold",
    )
    return figure


def _metric_figure(designs: dict[str, np.ndarray]) -> Any:
    names = list(designs)
    metrics = {name: _design_metrics(points) for name, points in designs.items()}
    keys = (
        ("min_pair", "minimum pairwise spacing", "larger is better"),
        ("mean_gap", "mean grid-to-design distance", "smaller is better"),
        ("max_gap", "worst-case grid-to-design distance", "smaller is better"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), constrained_layout=True)
    for axis, (key, title, subtitle) in zip(axes, keys, strict=True):
        values = [metrics[name][key] for name in names]
        bars = axis.bar(
            names,
            values,
            color=[METHOD_COLORS[name] for name in names],
            edgecolor="white",
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        axis.set_title(f"{title}\n({subtitle})")
        axis.tick_params(axis="x", rotation=35)
        axis.set_ylim(bottom=0.0)
        style_axis(axis)
    figure.suptitle("Coverage diagnostics for a fixed 20-point budget", fontsize=15, fontweight="bold")
    return figure


def _write_sampling_gif(
    path: Path,
    *,
    seed: int,
    n_points: int,
    frame_duration_ms: int,
    final_hold_ms: int,
    dpi: int,
) -> Path:
    """Write a slow, reproducible comparison of three sequential designs."""

    designs = _sampling_designs(seed, n_points)
    frames = [
        _sampling_frame(designs, count, dpi=dpi)
        for count in range(1, n_points + 1)
    ]
    durations = [frame_duration_ms] * len(frames)
    durations[-1] = final_hold_ms
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    return path


def _sampling_designs(seed: int, n_points: int) -> dict[str, np.ndarray]:
    random = np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n_points, 2))
    lhs = 2.0 * qmc.LatinHypercube(d=2, seed=seed + 1).random(n_points) - 1.0
    candidates = np.random.default_rng(seed + 2).uniform(
        -1.0, 1.0, size=(n_points * 80, 2)
    )
    maximin = _greedy_maximin(candidates, n_points, seed + 3)
    return {"Random": random, "LHS": lhs, "Maxmin": maximin}


def _sampling_frame(
    designs: dict[str, np.ndarray], count: int, *, dpi: int
) -> Image.Image:
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), constrained_layout=True)
    for axis, (name, points) in zip(axes, designs.items(), strict=True):
        axis.scatter(
            points[:count, 0],
            points[:count, 1],
            s=38,
            c=METHOD_COLORS[name],
            edgecolors="white",
            linewidths=0.7,
            alpha=0.95,
        )
        axis.set(
            title=name,
            xlim=(-1.05, 1.05),
            ylim=(-1.05, 1.05),
            xlabel="x₁",
            ylabel="x₂",
            aspect="equal",
        )
        style_axis(axis, grid=False)
        axis.set_facecolor("white")
    buffer = BytesIO()
    figure.savefig(buffer, dpi=dpi, format="png", facecolor=figure.get_facecolor())
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _design_metrics(points: np.ndarray) -> dict[str, float]:
    grid_axis = np.linspace(-1.0, 1.0, 80)
    mesh = np.meshgrid(grid_axis, grid_axis, indexing="xy")
    grid = np.column_stack([axis.ravel() for axis in mesh])
    distances = cdist(grid, points).min(axis=1)
    pairwise = pdist(points)
    return {
        "min_pair": float(np.min(pairwise)),
        "mean_gap": float(np.mean(distances)),
        "max_gap": float(np.max(distances)),
    }


__all__ = ["generate_doe_figures"]
