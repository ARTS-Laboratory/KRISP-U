"""Scientific toy fields, sequential states, and process visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from evaluation.fields.canonical.smooth import smooth_field
from evaluation.fields.synthetic_gp.kernel_fields import rough_multiscale_field
from evaluation.runners.design import initial_design, regular_grid
from evaluation.runners.sequential import SequentialState, run_sequential_design
from krispu.candidates import generate_candidates
from krispu.config import GPRConfig

from .styles import (
    BLUE,
    INK,
    ORANGE,
    RED,
    add_panel_label,
    save_figure,
    style_axis,
)


@dataclass(frozen=True)
class ToyRun:
    """One reproducible sequential KRISP-U run on a two-dimensional field."""

    name: str
    field: Any
    evaluation_points: np.ndarray
    states: tuple[SequentialState, ...]
    grid_size: int
    seed: int

    @property
    def final_state(self) -> SequentialState:
        return self.states[-1]


def build_toy_run(
    name: str,
    seed: int,
    *,
    final_budget: int = 20,
    grid_size: int = 48,
    candidate_count: int = 280,
    method: str = "support_adjusted_krispu",
) -> ToyRun:
    """Run a fixed-budget, fixed-seed toy experiment through the public runner."""

    if name == "smooth":
        field = smooth_field()
    elif name == "rough_multiscale":
        field = rough_multiscale_field(seed=seed)
    else:
        raise ValueError(f"Unknown toy field: {name}")
    initial_X, eligible = initial_design(
        "interior_maximin",
        field.domain,
        sample_count=5,
        boundary_margin=0.05,
        random_state=seed + 1,
        return_eligibility=True,
    )
    candidate_pool = generate_candidates(field.domain, candidate_count, "lhs", seed + 2)
    evaluation = regular_grid(field.domain, grid_size)
    config = GPRConfig(
        n_restarts_optimizer=0,
        random_state=seed,
        optimize_hyperparameters=True,
    )
    states = run_sequential_design(
        field.evaluate,
        field.domain,
        initial_X,
        candidate_pool,
        evaluation,
        method,
        final_budget,
        seed,
        field_name=name,
        trial=0,
        true_evaluation=field.evaluate(evaluation),
        gpr_config=config,
        initial_jackknife_eligible=eligible,
        selection_mode_label="KRISP-U" if method == "support_adjusted_krispu" else method,
    )
    return ToyRun(name, field, evaluation, tuple(states), grid_size, seed)


def build_toy_runs(
    seed: int,
    *,
    final_budget: int = 20,
    grid_size: int = 48,
    candidate_count: int = 280,
) -> dict[str, ToyRun]:
    """Build the smooth and rough/multiscale runs with separated seeds."""

    return {
        "smooth": build_toy_run(
            "smooth",
            seed + 11,
            final_budget=final_budget,
            grid_size=grid_size,
            candidate_count=candidate_count,
        ),
        "rough_multiscale": build_toy_run(
            "rough_multiscale",
            seed + 29,
            final_budget=final_budget,
            grid_size=grid_size,
            candidate_count=candidate_count,
        ),
    }


def write_toy_figures(run: ToyRun, output_dir: Path, *, dpi: int = 180) -> dict[str, Path]:
    """Write the truth, static snapshots, learning curve, and process GIF."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "truth": save_figure(_truth_figure(run), output_dir / f"{run.name}_field_truth.png", dpi=dpi),
        "snapshots": save_figure(
            _snapshot_figure(run), output_dir / f"{run.name}_snapshots.png", dpi=dpi
        ),
        "learning_curve": save_figure(
            _learning_curve_figure(run), output_dir / f"{run.name}_learning_curve.png", dpi=dpi
        ),
        "gif": _write_process_gif(run, output_dir / f"{run.name}_process.gif", dpi=dpi),
    }
    return paths


def _truth_figure(run: ToyRun) -> Any:
    state = run.final_state
    true = _field_array(state.true_field, run.grid_size)
    figure, axis = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    image = axis.imshow(true, extent=(-1, 1, -1, 1), origin="lower", cmap="viridis")
    _overlay_points(axis, state, show_order=True)
    axis.set(title=f"{_title(run.name)}: hidden response field", xlabel="x₁", ylabel="x₂")
    axis.set_aspect("equal")
    axis.set_facecolor("white")
    figure.colorbar(image, ax=axis, shrink=0.84, label="response")
    add_panel_label(axis, "A")
    return figure


def _snapshot_figure(run: ToyRun) -> Any:
    states = _snapshot_states(run.states)
    figure, axes = plt.subplots(2, len(states), figsize=(3.0 * len(states), 6.5), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, len(states))
    true_limits = _shared_limits([state.true_field for state in run.states])
    uncertainty_max = max(
        float(np.max(_uncertainty(state))) for state in run.states
    )
    for column, state in enumerate(states):
        predicted = _field_array(state.predicted_field, run.grid_size)
        uncertainty = _field_array(_uncertainty(state), run.grid_size)
        top = axes[0, column]
        bottom = axes[1, column]
        top.imshow(
            predicted,
            extent=(-1, 1, -1, 1),
            origin="lower",
            cmap="viridis",
            vmin=true_limits[0],
            vmax=true_limits[1],
        )
        bottom.imshow(
            uncertainty,
            extent=(-1, 1, -1, 1),
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=uncertainty_max,
        )
        for axis in (top, bottom):
            _overlay_points(axis, state)
            axis.set(xlim=(-1, 1), ylim=(-1, 1), xticks=[], yticks=[])
            axis.set_aspect("equal")
        top.set_title(f"n = {state.sample_count}")
        if column == 0:
            top.set_ylabel("reconstruction")
            bottom.set_ylabel("KRISP-U uncertainty")
        else:
            top.set_ylabel("")
            bottom.set_ylabel("")
    figure.suptitle(f"{_title(run.name)}: reconstruction closes the gaps", fontsize=15, fontweight="bold")
    return figure


def _learning_curve_figure(run: ToyRun) -> Any:
    counts = np.asarray([state.sample_count for state in run.states])
    nrmse = np.asarray([state.metrics.nrmse for state in run.states])
    uncertainty = np.asarray([np.mean(_uncertainty(state)) for state in run.states])
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].plot(counts, nrmse, "o-", color=ORANGE, linewidth=2, label="field NRMSE")
    axes[0].set(xlabel="number of measurements", ylabel="NRMSE", title="Reconstruction error")
    axes[0].set_ylim(bottom=0.0)
    axes[1].plot(counts, uncertainty, "o-", color=BLUE, linewidth=2, label="mean uncertainty")
    axes[1].set(
        xlabel="number of measurements",
        ylabel="mean KRISP-U uncertainty",
        title="Uncertainty contracts as support grows",
    )
    axes[1].set_ylim(bottom=0.0)
    for axis in axes:
        style_axis(axis)
        axis.legend(loc="best")
    figure.suptitle(f"{_title(run.name)}: sequential learning", fontsize=15, fontweight="bold")
    return figure


def _write_process_gif(run: ToyRun, path: Path, *, dpi: int) -> Path:
    frames = [_state_frame(run, state, dpi=dpi) for state in run.states]
    durations = [450] * len(frames)
    durations[-1] = 1400
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


def _state_frame(run: ToyRun, state: SequentialState, *, dpi: int) -> Image.Image:
    true = _field_array(state.true_field, run.grid_size)
    predicted = _field_array(state.predicted_field, run.grid_size)
    uncertainty = _field_array(_uncertainty(state), run.grid_size)
    error = np.abs(true - predicted)
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 7.3), constrained_layout=True)
    axes = axes.ravel()
    fields = (true, predicted, uncertainty, error)
    labels = ("hidden field", "current reconstruction", "KRISP-U uncertainty", "absolute error")
    cmaps = ("viridis", "viridis", "magma", "inferno")
    limits = (
        _shared_limits([item.true_field for item in run.states]),
        _shared_limits([item.predicted_field for item in run.states]),
        (0.0, max(float(np.max(_uncertainty(item))) for item in run.states)),
        (0.0, max(float(np.max(np.abs(item.true_field - item.predicted_field))) for item in run.states)),
    )
    for axis, values, label, cmap, (low, high) in zip(axes, fields, labels, cmaps, limits, strict=True):
        image = axis.imshow(
            values,
            extent=(-1, 1, -1, 1),
            origin="lower",
            cmap=cmap,
            vmin=low,
            vmax=high,
        )
        _overlay_points(axis, state, show_order=True)
        axis.set(title=label, xlim=(-1, 1), ylim=(-1, 1), aspect="equal")
        axis.set_xlabel("x₁")
        axis.set_ylabel("x₂")
        figure.colorbar(image, ax=axis, shrink=0.78)
    figure.suptitle(f"{_title(run.name)} | n = {state.sample_count}", fontsize=15, fontweight="bold")
    buffer = BytesIO()
    figure.savefig(buffer, dpi=dpi, format="png", facecolor=figure.get_facecolor())
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _overlay_points(axis: Any, state: SequentialState, *, show_order: bool = False) -> None:
    points = np.asarray(state.observed_X)
    axis.scatter(points[:, 0], points[:, 1], s=22, c="white", edgecolors=INK, linewidths=0.7)
    if show_order:
        for index, point in enumerate(points):
            axis.text(point[0] + 0.025, point[1] + 0.025, str(index + 1), fontsize=7, color=INK)
    if state.recommended_point is not None:
        axis.scatter(
            *state.recommended_point,
            marker="*",
            s=115,
            c=RED,
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )


def _snapshot_states(states: tuple[SequentialState, ...] | list[SequentialState]) -> list[SequentialState]:
    values = list(states)
    indices = np.linspace(0, len(values) - 1, min(4, len(values)), dtype=int)
    return [values[int(index)] for index in dict.fromkeys(indices)]


def _field_array(values: np.ndarray, grid_size: int) -> np.ndarray:
    return np.asarray(values).reshape(grid_size, grid_size)


def _uncertainty(state: SequentialState) -> np.ndarray:
    if state.krispu_uncertainty is not None:
        return np.asarray(state.krispu_uncertainty)
    if state.posterior_std is not None:
        return np.asarray(state.posterior_std)
    raise ValueError("A toy state must contain uncertainty values.")


def _shared_limits(values: list[np.ndarray] | tuple[np.ndarray, ...]) -> tuple[float, float]:
    flat = np.concatenate([np.asarray(value).reshape(-1) for value in values])
    low, high = float(np.min(flat)), float(np.max(flat))
    if np.isclose(low, high):
        return low - 1.0e-6, high + 1.0e-6
    return low, high


def _title(name: str) -> str:
    return "Smooth toy problem" if name == "smooth" else "Rough / multiscale toy problem"


__all__ = ["ToyRun", "build_toy_run", "build_toy_runs", "write_toy_figures"]
