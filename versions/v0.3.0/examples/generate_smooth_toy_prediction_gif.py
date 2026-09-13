"""Create a smooth sequential prediction/uncertainty slide figure."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from PIL import Image
from scipy.ndimage import label

from krispu import GPRConfig, KrispURecommender, ObservationSet
from krispu.candidates import valid_candidate_mask
from krispu.domains import ContinuousDomain

FRAME_DURATION_MS = 1_400
GRID_SIZE = 101
CANDIDATE_GRID_SIZE = 41
INITIAL_N = 5
FINAL_N = 25
RANDOM_STATE = 11

VERSION_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = VERSION_ROOT / "slide_figures" / "smooth_toy_prediction_uncertainty.gif"
TRACE_PATH = OUTPUT_PATH.with_suffix(".csv")
SUMMARY_PLOT_PATH = VERSION_ROOT / "slide_figures" / "smooth_toy_uncertainty_r2.png"

DOMAIN = ContinuousDomain([[-1.0, 1.0], [-1.0, 1.0]], names=("x", "y"))


@dataclass(frozen=True)
class FrameState:
    """One sequential model state for the animation."""

    n: int
    X: np.ndarray
    prediction: np.ndarray
    uncertainty: np.ndarray
    next_point: np.ndarray | None


def smooth_field(points: np.ndarray) -> np.ndarray:
    """Smooth multi-feature toy response."""

    x, y = points[:, 0], points[:, 1]
    return (
        0.42 * x
        - 0.22 * y
        + 0.30 * np.sin(1.6 * x + 0.25 * y)
        - 0.22 * np.cos(1.4 * y - 0.20 * x)
        + 0.35 * np.exp(-3.5 * ((x - 0.35) ** 2 + (y + 0.25) ** 2))
        - 0.22 * np.exp(-4.5 * ((x + 0.50) ** 2 + (y - 0.45) ** 2))
        + 0.10 * np.exp(-6.0 * ((x + 0.35) ** 2 + (y + 0.55) ** 2))
        + 0.22 * np.sin(2.6 * x * y)
        + 0.18 * np.exp(-10.0 * ((x - 0.65) ** 2 + (y - 0.55) ** 2))
        - 0.16 * np.exp(-10.0 * ((x + 0.70) ** 2 + (y - 0.10) ** 2))
    )


def main() -> None:
    X = _initial_seed()
    y = smooth_field(X)

    grid_axis = np.linspace(-1.0, 1.0, GRID_SIZE)
    grid_x, grid_y = np.meshgrid(grid_axis, grid_axis)
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    underlying_field = smooth_field(grid).reshape(GRID_SIZE, GRID_SIZE)

    candidate_axis = np.linspace(-1.0, 1.0, CANDIDATE_GRID_SIZE)
    candidate_x, candidate_y = np.meshgrid(candidate_axis, candidate_axis)
    candidate_grid = np.column_stack((candidate_x.ravel(), candidate_y.ravel()))
    config = GPRConfig(
        n_restarts_optimizer=0,
        random_state=RANDOM_STATE,
        length_scale_initial=0.35,
        length_scale_bounds=(0.02, 20.0),
    )
    states: list[FrameState] = []
    trace_rows: list[dict[str, object]] = []
    references = grid
    for n in range(INITIAL_N, FINAL_N + 1):
        observations = ObservationSet(X, y)
        recommender = KrispURecommender(
            DOMAIN,
            uncertainty="krispu_loo",
            gpr_config=config,
            random_state=RANDOM_STATE,
            n_candidates=len(candidate_grid),
            min_normalized_distance=0.06,
        )
        diagnostics = recommender.evaluate_uncertainty(observations, references)
        prediction = diagnostics.predicted_mean
        uncertainty = diagnostics.loo_field_uncertainty
        next_point = None
        if n < FINAL_N:
            next_point = _centroid_next_point(
                grid,
                X,
                uncertainty,
                candidate_grid,
            )
        states.append(
            FrameState(
                n=n,
                X=X.copy(),
                prediction=prediction.reshape(GRID_SIZE, GRID_SIZE),
                uncertainty=uncertainty.reshape(GRID_SIZE, GRID_SIZE),
                next_point=None if next_point is None else next_point.copy(),
            )
        )
        trace_rows.append(
            {
                "n_points": n,
                "mean_krisp_uncertainty": float(np.mean(uncertainty)),
                "max_krisp_uncertainty": float(np.max(uncertainty)),
                "next_x": "" if next_point is None else float(next_point[0]),
                "next_y": "" if next_point is None else float(next_point[1]),
            }
        )
        if next_point is not None:
            X = np.vstack((X, next_point))
            y = np.append(y, smooth_field(next_point.reshape(1, -1))[0])

    prediction_min = min(float(np.min(state.prediction)) for state in states)
    prediction_max = max(float(np.max(state.prediction)) for state in states)
    prediction_pad = max((prediction_max - prediction_min) * 0.04, 1e-12)
    prediction_limits = (prediction_min - prediction_pad, prediction_max + prediction_pad)
    true_values = underlying_field.reshape(-1)
    total_variation = float(np.sum((true_values - np.mean(true_values)) ** 2))
    r2_values = [
        1.0
        - float(np.sum((state.prediction.reshape(-1) - true_values) ** 2)) / total_variation
        for state in states
    ]
    for row, r2 in zip(trace_rows, r2_values, strict=True):
        row["r2"] = r2

    images = [
        _draw_frame(
            state,
            grid_x,
            grid_y,
            underlying_field,
            prediction_limits,
        )
        for state in states
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=images[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=False,
    )
    with TRACE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
        writer.writeheader()
        writer.writerows(trace_rows)
    _save_summary_plot(states, r2_values)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {TRACE_PATH}")
    print(f"Wrote {SUMMARY_PLOT_PATH}")


def _initial_seed() -> np.ndarray:
    """Return four edge seeds plus one reproducibly offset center seed."""

    rng = np.random.default_rng(RANDOM_STATE)
    center = rng.normal(loc=0.0, scale=0.08, size=2)
    center = np.clip(center, -0.18, 0.18)
    return np.vstack(
        (
            np.array(
                [
                    [-1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, -1.0],
                    [0.0, 1.0],
                ],
                dtype=float,
            ),
            center,
        )
    )


def _centroid_next_point(
    grid: np.ndarray,
    X: np.ndarray,
    uncertainty: np.ndarray,
    candidate_grid: np.ndarray,
) -> np.ndarray:
    """Pick the weighted centroid of the strongest connected LOO region."""

    weights = np.maximum(uncertainty.reshape(-1), 0.0)
    threshold = float(np.quantile(weights, 0.75))
    high_uncertainty = weights.reshape(GRID_SIZE, GRID_SIZE) >= threshold
    regions, n_regions = label(high_uncertainty, structure=np.ones((3, 3), dtype=int))
    flat_regions = regions.reshape(-1)
    region_scores = [
        (float(np.sum(weights[flat_regions == region_index])), region_index)
        for region_index in range(1, n_regions + 1)
    ]
    if not region_scores:
        raise ValueError("The LOO uncertainty field has no connected high-uncertainty region.")

    _, selected_region = max(region_scores)
    selected_weights = np.where(flat_regions == selected_region, weights, 0.0)
    total_weight = float(np.sum(selected_weights))
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        raise FloatingPointError("The selected LOO uncertainty region has no finite weight.")
    centroid = np.sum(grid * selected_weights[:, None], axis=0) / total_weight

    normalized_centroid = DOMAIN.normalize(centroid)
    normalized_observations = DOMAIN.normalize(X)
    if np.min(
        np.linalg.norm(normalized_observations - normalized_centroid, axis=1)
    ) >= 0.06:
        return centroid

    eligible = valid_candidate_mask(
        DOMAIN,
        candidate_grid,
        X,
        minimum_normalized_distance=0.06,
    )
    if not np.any(eligible):
        raise ValueError("The candidate grid contains no valid fallback point.")
    eligible_grid = candidate_grid[eligible]
    nearest = int(np.argmin(np.linalg.norm(eligible_grid - centroid, axis=1)))
    return eligible_grid[nearest].copy()


def _save_summary_plot(states: list[FrameState], r2_values: list[float]) -> None:
    """Save uncertainty and reconstruction R-squared against sample count."""

    sample_counts = np.asarray([state.n for state in states])
    mean_uncertainty = np.asarray([np.mean(state.uncertainty) for state in states])
    max_uncertainty = np.asarray([np.max(state.uncertainty) for state in states])

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].plot(sample_counts, mean_uncertainty, marker="o", label="mean")
    axes[0].plot(
        sample_counts,
        max_uncertainty,
        marker="s",
        linestyle="--",
        label="maximum",
    )
    axes[0].set(
        title="KRISP-U Uncertainty",
        xlabel="points added",
        ylabel="LOO field uncertainty",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[1].plot(sample_counts, r2_values, marker="o", color="darkgreen")
    axes[1].set(title="Reconstruction Accuracy", xlabel="points added", ylabel="$R^2$")
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    axes[1].grid(alpha=0.25)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    figure.savefig(SUMMARY_PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _draw_frame(
    state: FrameState,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    underlying_field: np.ndarray,
    prediction_limits: tuple[float, float],
) -> Image.Image:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    underlying_plot = axes[0].pcolormesh(
        grid_x,
        grid_y,
        underlying_field,
        shading="auto",
        cmap="viridis",
        vmin=prediction_limits[0],
        vmax=prediction_limits[1],
    )
    figure.colorbar(
        underlying_plot,
        ax=axes[0],
        label="underlying response",
        shrink=0.60,
        pad=0.025,
        aspect=24,
    )
    axes[0].set_title("Underlying Domain")
    _plot_points(axes[0], state.X, state.next_point)

    prediction_plot = axes[1].pcolormesh(
        grid_x,
        grid_y,
        state.prediction,
        shading="auto",
        cmap="viridis",
        vmin=prediction_limits[0],
        vmax=prediction_limits[1],
    )
    figure.colorbar(
        prediction_plot,
        ax=axes[1],
        label="kriging / GPR predicted response",
        shrink=0.60,
        pad=0.025,
        aspect=24,
    )
    axes[1].set_title(f"Kriging Surrogate Model (n={state.n})")
    _plot_points(axes[1], state.X, state.next_point)

    frame_uncertainty_max = float(np.max(state.uncertainty))
    if frame_uncertainty_max <= 0.0 or not np.isfinite(frame_uncertainty_max):
        raise FloatingPointError("The frame LOO uncertainty must have a positive finite maximum.")
    normalized_uncertainty = state.uncertainty / frame_uncertainty_max
    uncertainty_plot = axes[2].pcolormesh(
        grid_x,
        grid_y,
        normalized_uncertainty,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    figure.colorbar(
        uncertainty_plot,
        ax=axes[2],
        label="normalized KRISP-U LOO uncertainty",
        shrink=0.60,
        pad=0.025,
        aspect=24,
    )
    axes[2].set_title("KRISP-U Uncertainty")
    _plot_points(axes[2], state.X, state.next_point)

    for axis in axes:
        axis.set(
            xlim=(-1.0, 1.0),
            ylim=(-1.0, 1.0),
            aspect="equal",
            xlabel="x",
            ylabel="y",
        )
    return _figure_to_image(figure)


def _plot_points(axis: plt.Axes, X: np.ndarray, next_point: np.ndarray | None) -> None:
    axis.scatter(
        X[:, 0],
        X[:, 1],
        c="white",
        edgecolors="black",
        s=48,
        linewidth=0.7,
        label="measured",
        zorder=4,
    )
    if next_point is not None:
        axis.scatter(
            next_point[0],
            next_point[1],
            c="gold",
            edgecolors="black",
            marker="*",
            s=200,
            linewidths=1.0,
            label="next point",
            zorder=5,
            clip_on=False,
        )
    axis.legend(loc="upper left", fontsize=8)


def _figure_to_image(figure: plt.Figure) -> Image.Image:
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    image = Image.fromarray(
        rgba.reshape(height, width, 4)
    ).convert("P", palette=Image.Palette.ADAPTIVE)
    plt.close(figure)
    return image


if __name__ == "__main__":
    main()
