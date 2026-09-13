"""Visual explanations of the KRISP-U uncertainty construction."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from krispu.config import GPRConfig
from krispu.domains import ContinuousDomain
from krispu.surrogates.gpr import GPRSurrogate

from .styles import INK, ORANGE, RED, add_panel_label, save_figure
from .toy_figures import ToyRun, build_toy_run


def generate_theory_figures(
    output_dir: Path,
    *,
    state: Any | None = None,
    run: ToyRun | None = None,
    seed: int = 202603,
    jackknife_sample_count: int = 10,
    jackknife_frame_ms: int = 650,
    jackknife_final_hold_ms: int = 2500,
    dpi: int = 180,
) -> dict[str, Path]:
    """Write LOO sensitivity, kernel support, and final decomposition figures."""

    reference_run = run
    if state is None:
        reference_run = run or build_toy_run(
            "smooth", seed + 41, final_budget=16, grid_size=48, candidate_count=220
        )
        state = reference_run.states[-1]
    elif reference_run is None:
        reference_run = _run_from_state(state, seed)
    grid_size = _grid_size(state)
    paths = {
        "loo_sensitivity": save_figure(
            _loo_figure(state, grid_size), output_dir / "loo_sensitivity.png", dpi=dpi
        ),
        "kernel_support": save_figure(
            _support_figure(state, grid_size), output_dir / "kernel_support.png", dpi=dpi
        ),
        "uncertainty_decomposition": save_figure(
            _decomposition_figure(state, grid_size),
            output_dir / "uncertainty_decomposition.png",
            dpi=dpi,
        ),
        "jackknife_process": _write_jackknife_gif(
            reference_run,
            output_dir / "jackknife_process.gif",
            sample_count=jackknife_sample_count,
            frame_duration_ms=jackknife_frame_ms,
            final_hold_ms=jackknife_final_hold_ms,
            dpi=dpi,
        ),
    }
    return paths


def _loo_figure(state: Any, grid_size: int) -> Any:
    values = _required(state.jackknife_field_sensitivity, "LOO sensitivity")
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), constrained_layout=True)
    image = _map(axes[0], values, grid_size, "magma", "buffered-jackknife sensitivity S(x)", state)
    axes[0].figure.colorbar(image, ax=axes[0], shrink=0.82, label="S(x)")
    axes[1].hist(values, bins=32, color=ORANGE, alpha=0.88, edgecolor="white")
    axes[1].set(
        title="Influence is concentrated near\npoorly reconstructed features",
        xlabel="S(x)",
        ylabel="evaluation locations",
    )
    axes[1].axvline(float(np.mean(values)), color=INK, linestyle="--", linewidth=1.2, label="mean")
    axes[1].legend()
    figure.suptitle(
        f"Leave-one-out field sensitivity | n = {state.sample_count}",
        fontsize=15,
        fontweight="bold",
    )
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    return figure


def _support_figure(state: Any, grid_size: int) -> Any:
    deficit = _required(state.kernel_support_deficit, "kernel support deficit")
    correlation = _required(
        state.maximum_kernel_correlation_to_observations,
        "maximum kernel correlation",
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), constrained_layout=True)
    image = _map(axes[0], deficit, grid_size, "magma", "kernel support deficit D(x)", state)
    axes[0].figure.colorbar(image, ax=axes[0], shrink=0.82, label="D(x)")
    image = _map(
        axes[1],
        correlation,
        grid_size,
        "viridis",
        "maximum support correlation",
        state,
        limits=(0.0, 1.0),
    )
    axes[1].figure.colorbar(image, ax=axes[1], shrink=0.82, label="max corrₖ")
    figure.suptitle(
        "Kernel support: extrapolation is visible even when the GP is smooth",
        fontsize=15,
        fontweight="bold",
    )
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    return figure


def _decomposition_figure(state: Any, grid_size: int) -> Any:
    sensitivity = _required(state.jackknife_field_sensitivity, "LOO sensitivity")
    deficit = _required(state.kernel_support_deficit, "kernel support deficit")
    uncertainty = _required(state.krispu_uncertainty, "KRISP-U uncertainty")
    figure, axes = plt.subplots(1, 4, figsize=(15.2, 4.3), constrained_layout=True)
    maps = (
        (sensitivity, "magma", "field sensitivity S(x)", "S"),
        (deficit, "magma", "support deficit D(x)", "D"),
        (np.sqrt(deficit), "plasma", "support weight √D(x)", "√D"),
        (uncertainty, "inferno", "final KRISP-U U(x)", "U"),
    )
    for axis, (values, cmap, title, label) in zip(axes, maps, strict=True):
        image = _map(axis, values, grid_size, cmap, title, state)
        axis.figure.colorbar(image, ax=axis, shrink=0.82, label=label)
    figure.suptitle(
        "Final uncertainty decomposition:  U(x) = S(x) × √D(x)",
        fontsize=15,
        fontweight="bold",
    )
    for axis, label in zip(axes, ("A", "B", "C", "D"), strict=True):
        add_panel_label(axis, label)
    return figure


def _write_jackknife_gif(
    run: ToyRun,
    path: Path,
    *,
    sample_count: int,
    frame_duration_ms: int,
    final_hold_ms: int,
    dpi: int,
) -> Path:
    """Animate true single-point LOO submodels and cumulative uncertainty."""

    state = _state_at_sample_count(run, sample_count)
    fold_means, uncertainty = _loo_animation_fields(state, run.seed)
    support_deficit = np.asarray(state.kernel_support_deficit, dtype=float)
    display_uncertainty = uncertainty * np.sqrt(np.maximum(support_deficit[:, None], 0.0))
    recommended_point = np.asarray(state.recommended_point, dtype=float)
    frames = [
        _jackknife_frame(
            state,
            fold_means,
            display_uncertainty,
            fold_index,
            recommended_point,
            dpi=dpi,
        )
        for fold_index in range(fold_means.shape[1])
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


def _loo_animation_fields(
    state: Any,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit one fixed-kernel kriging submodel for each omitted observation."""

    domain = ContinuousDomain([[-1.0, 1.0], [-1.0, 1.0]])
    X = domain.normalize(np.asarray(state.observed_X, dtype=float))
    y = np.asarray(state.observed_y, dtype=float)
    reference = domain.normalize(np.asarray(state.evaluation_points, dtype=float))
    config = GPRConfig(
        n_restarts_optimizer=0,
        random_state=seed,
        optimize_hyperparameters=True,
    )
    full = GPRSurrogate(config).fit(X, y)
    fold_means = np.empty((len(reference), len(X)), dtype=float)
    for fold_index in range(len(X)):
        keep = np.arange(len(X)) != fold_index
        fold = GPRSurrogate(config).fit_fixed_kernel(
            X[keep],
            y[keep],
            frozen_kernel=full.frozen_kernel,
        )
        fold_means[:, fold_index], _ = fold.predict(reference)

    uncertainty = np.empty_like(fold_means)
    for count in range(1, len(X) + 1):
        values = fold_means[:, :count]
        mean = np.mean(values, axis=1)
        uncertainty[:, count - 1] = np.sqrt(
            (count - 1.0) / count * np.sum((values - mean[:, None]) ** 2, axis=1)
        )
    return fold_means, uncertainty


def _jackknife_frame(
    state: Any,
    fold_means: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
    fold_index: int,
    recommended_point: NDArray[np.float64] | None,
    *,
    dpi: int,
) -> Image.Image:
    grid_size = _grid_size(state)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), constrained_layout=True)
    mean_values = fold_means[:, fold_index].reshape(grid_size, grid_size)
    uncertainty_values = uncertainty[:, fold_index].reshape(grid_size, grid_size)
    mean_limits = (float(np.min(fold_means)), float(np.max(fold_means)))
    uncertainty_limit = max(float(np.max(uncertainty)), np.finfo(float).eps)
    images = (
        axes[0].imshow(
            mean_values,
            extent=(-1, 1, -1, 1),
            origin="lower",
            cmap="viridis",
            vmin=mean_limits[0],
            vmax=mean_limits[1],
        ),
        axes[1].imshow(
            uncertainty_values,
            extent=(-1, 1, -1, 1),
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=uncertainty_limit,
        ),
    )
    axes[0].figure.colorbar(
        images[0], ax=axes[0], shrink=0.68, fraction=0.045, pad=0.025, label="predicted response"
    )
    axes[1].figure.colorbar(
        images[1],
        ax=axes[1],
        shrink=0.68,
        fraction=0.045,
        pad=0.025,
        label="support-adjusted LOO uncertainty",
    )
    observed = np.asarray(state.observed_X)
    retained = np.delete(observed, fold_index, axis=0)
    for axis in axes:
        axis.scatter(
            retained[:, 0],
            retained[:, 1],
            s=22,
            c="white",
            edgecolors=INK,
            linewidths=0.7,
            zorder=3,
        )
        axis.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), xlabel="x₁", ylabel="x₂", aspect="equal")
    axes[0].set_title(f"LOO submodel · omit {fold_index + 1}")
    axes[1].set_title(f"Support-adjusted uncertainty · {fold_index + 1}/{len(observed)}")
    if fold_index == len(observed) - 1 and recommended_point is not None:
        for axis in axes:
            axis.scatter(
                *recommended_point,
                marker="*",
                s=145,
                c=RED,
                edgecolors="white",
                linewidths=0.9,
                zorder=5,
            )
            axis.text(
                recommended_point[0] - 0.03,
                recommended_point[1] + 0.04,
                "next point",
                color=RED,
                fontsize=9,
                fontweight="bold",
                ha="right",
                zorder=6,
            )
    buffer = BytesIO()
    figure.savefig(buffer, dpi=dpi, format="png", facecolor=figure.get_facecolor())
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _run_from_state(state: Any, seed: int) -> ToyRun:
    """Build a smooth fallback run when only a state was supplied."""

    return build_toy_run(
        "smooth",
        seed,
        final_budget=state.sample_count,
        grid_size=_grid_size(state),
        candidate_count=max(160, state.sample_count * 10),
    )


def _state_at_sample_count(run: ToyRun, sample_count: int) -> Any:
    for state in run.states:
        if state.sample_count == sample_count:
            return state
    raise ValueError(f"The run does not contain a state with {sample_count} observations.")


def _map(
    axis: Any,
    values: np.ndarray,
    grid_size: int,
    cmap: str,
    title: str,
    state: Any,
    limits: tuple[float, float] | None = None,
) -> Any:
    array = np.asarray(values).reshape(grid_size, grid_size)
    if limits is None:
        limits = (float(np.min(array)), float(np.max(array)))
    image = axis.imshow(
        array,
        extent=(-1, 1, -1, 1),
        origin="lower",
        cmap=cmap,
        vmin=limits[0],
        vmax=max(limits[1], limits[0] + np.finfo(float).eps),
    )
    points = np.asarray(state.observed_X)
    axis.scatter(points[:, 0], points[:, 1], c="white", edgecolors=INK, s=23, linewidths=0.7)
    if state.recommended_point is not None:
        axis.scatter(
            *state.recommended_point,
            c=RED,
            marker="*",
            s=105,
            edgecolors="white",
            linewidths=0.8,
        )
    axis.set(title=title, xlim=(-1, 1), ylim=(-1, 1), xlabel="x₁", ylabel="x₂", aspect="equal")
    return image


def _required(value: Any, label: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"The reference state does not contain {label}.")
    return np.asarray(value, dtype=float)


def _grid_size(state: Any) -> int:
    count = len(np.asarray(state.evaluation_points))
    size = round(np.sqrt(count))
    if size * size != count:
        raise ValueError("Theory figures require a square evaluation grid.")
    return size


__all__ = ["generate_theory_figures"]
