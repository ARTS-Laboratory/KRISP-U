"""Create a slow, side-by-side kriging and jackknife LOO demonstration.

The left panel shows the kriging/GPR field for the current leave-one-out
fold.  The right panel shows the jackknife field uncertainty computed from
the LOO fields revealed so far.  The folds are accumulated for display only:
each fold still removes one point from the complete data set independently.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from krispu import GPRConfig, ObservationSet
from krispu.domains import ContinuousDomain
from krispu.surrogates import GPRSurrogate
from krispu.uncertainty import compute_bruteforce_loo, jackknife_std

# Editable demonstration settings.
FRAME_DURATION_MS = 1_400
GRID_SIZE = 65
RANDOM_STATE = 7

VERSION_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = VERSION_ROOT / "slide_figures" / "jackknife_kriging.gif"
TRACE_PATH = OUTPUT_PATH.with_suffix(".csv")

DOMAIN = ContinuousDomain([[-1.0, 1.0], [-1.0, 1.0]], names=("x", "y"))


def demo_field(points: np.ndarray) -> np.ndarray:
    """Smooth synthetic response used to make the procedure easy to see."""

    x, y = points[:, 0], points[:, 1]
    return (
        0.6 * np.sin(2.5 * x)
        - 0.4 * np.cos(2.0 * y)
        - 0.25 * x * y
        - 0.5 * np.exp(-3.0 * ((x - 0.35) ** 2 + (y + 0.25) ** 2))
        + 0.24 * np.exp(-8.0 * ((x + 0.55) ** 2 + (y - 0.42) ** 2))
        - 0.18 * np.exp(-12.0 * ((x + 0.45) ** 2 + (y + 0.55) ** 2))
        + 0.12 * np.sin(4.0 * x * y)
    )


def main() -> None:
    X = np.array(
        [
            [-0.98, -0.85],
            [0.96, -0.20],
            [-0.15, 0.97],
            [-0.65, -0.45],
            [-0.35, -0.10],
            [0.05, -0.50],
            [0.40, -0.42],
            [0.65, 0.05],
            [-0.60, 0.35],
            [-0.15, 0.45],
            [0.30, 0.55],
            [0.55, 0.40],
        ],
        dtype=float,
    )
    y = demo_field(X)
    observations = ObservationSet(X, y)

    grid_axis = np.linspace(-1.0, 1.0, GRID_SIZE)
    grid_x, grid_y = np.meshgrid(grid_axis, grid_axis)
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    config = GPRConfig(
        n_restarts_optimizer=0,
        random_state=RANDOM_STATE,
        # Keep the demonstration fit away from the default upper bound.
        length_scale_bounds=(0.02, 5.0),
    )
    complete = GPRSurrogate(config).fit(DOMAIN.normalize(X), y)
    complete_mean, _ = complete.predict(DOMAIN.normalize(grid))
    loo = compute_bruteforce_loo(
        complete,
        observations,
        reference_points_normalized=DOMAIN.normalize(grid),
    )

    loo_means = loo.field_means.reshape(GRID_SIZE, GRID_SIZE, -1)
    jackknife_fields = [
        jackknife_std(loo.field_means[:, :count])[1].reshape(GRID_SIZE, GRID_SIZE)
        for count in range(1, loo.field_means.shape[1] + 1)
    ]
    prediction_min = float(
        min(np.min(complete_mean), *(np.min(field) for field in loo_means.transpose(2, 0, 1)))
    )
    prediction_max = float(
        max(np.max(complete_mean), *(np.max(field) for field in loo_means.transpose(2, 0, 1)))
    )
    response_pad = max((prediction_max - prediction_min) * 0.04, 1e-12)
    prediction_limits = (prediction_min - response_pad, prediction_max + response_pad)
    uncertainty_max = max(float(np.max(jackknife_fields[-1])), 1e-12)
    recommended_point = _recommended_point(grid, X, jackknife_fields[-1])

    frames = [_draw_frame(
        X=X,
        grid_x=grid_x,
        grid_y=grid_y,
        current_mean=complete_mean.reshape(GRID_SIZE, GRID_SIZE),
        removed_index=None,
        cumulative_uncertainty=np.zeros((GRID_SIZE, GRID_SIZE)),
        cumulative_count=0,
        total_count=len(loo.loo_eligible_indices),
        prediction_limits=prediction_limits,
        uncertainty_max=uncertainty_max,
    )]
    trace_rows = [
        {
            "frame": 0,
            "removed_index": "",
            "removed_x": "",
            "removed_y": "",
            "cumulative_loo_folds": 0,
            "total_loo_folds": len(loo.loo_eligible_indices),
            "mean_jackknife_uncertainty": 0.0,
            "max_jackknife_uncertainty": 0.0,
            "recommended_x": "",
            "recommended_y": "",
        }
    ]

    for count, removed_index in enumerate(loo.loo_eligible_indices, start=1):
        uncertainty = jackknife_fields[count - 1]
        frames.append(_draw_frame(
            X=X,
            grid_x=grid_x,
            grid_y=grid_y,
            current_mean=loo_means[:, :, count - 1],
            removed_index=int(removed_index),
            cumulative_uncertainty=uncertainty,
            cumulative_count=count,
            total_count=len(loo.loo_eligible_indices),
            prediction_limits=prediction_limits,
            uncertainty_max=uncertainty_max,
        ))
        trace_rows.append(
            {
                "frame": count,
                "removed_index": int(removed_index),
                "removed_x": float(X[removed_index, 0]),
                "removed_y": float(X[removed_index, 1]),
                "cumulative_loo_folds": count,
                "total_loo_folds": len(loo.loo_eligible_indices),
                "mean_jackknife_uncertainty": float(np.mean(uncertainty)),
                "max_jackknife_uncertainty": float(np.max(uncertainty)),
                "recommended_x": "",
                "recommended_y": "",
            }
        )

    final_removed_index = int(loo.loo_eligible_indices[-1])
    frames.append(_draw_frame(
        X=X,
        grid_x=grid_x,
        grid_y=grid_y,
        current_mean=loo_means[:, :, -1],
        removed_index=final_removed_index,
        recommended_point=recommended_point,
        cumulative_uncertainty=jackknife_fields[-1],
        cumulative_count=len(loo.loo_eligible_indices),
        total_count=len(loo.loo_eligible_indices),
        prediction_limits=prediction_limits,
        uncertainty_max=uncertainty_max,
    ))
    trace_rows.append(
        {
            "frame": len(trace_rows),
            "removed_index": final_removed_index,
            "removed_x": float(X[final_removed_index, 0]),
            "removed_y": float(X[final_removed_index, 1]),
            "cumulative_loo_folds": len(loo.loo_eligible_indices),
            "total_loo_folds": len(loo.loo_eligible_indices),
            "mean_jackknife_uncertainty": float(np.mean(jackknife_fields[-1])),
            "max_jackknife_uncertainty": float(np.max(jackknife_fields[-1])),
            "recommended_x": float(recommended_point[0]),
            "recommended_y": float(recommended_point[1]),
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=False,
    )
    with TRACE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
        writer.writeheader()
        writer.writerows(trace_rows)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {TRACE_PATH}")


def _draw_frame(
    *,
    X: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    current_mean: np.ndarray,
    removed_index: int | None,
    cumulative_uncertainty: np.ndarray,
    cumulative_count: int,
    total_count: int,
    prediction_limits: tuple[float, float],
    uncertainty_max: float,
    recommended_point: np.ndarray | None = None,
) -> Image.Image:
    _ = cumulative_count, total_count
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)

    mean_plot = axes[0].pcolormesh(
        grid_x,
        grid_y,
        current_mean,
        shading="auto",
        cmap="viridis",
        vmin=prediction_limits[0],
        vmax=prediction_limits[1],
    )
    figure.colorbar(
        mean_plot,
        ax=axes[0],
        label="kriging / GPR predicted response",
        shrink=0.68,
        pad=0.025,
        aspect=24,
    )
    axes[0].set_title("Kriging Surrogate Model (n-1)")
    _plot_points(axes[0], X, removed_index, recommended_point)

    uncertainty_plot = axes[1].pcolormesh(
        grid_x,
        grid_y,
        cumulative_uncertainty,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=uncertainty_max,
    )
    figure.colorbar(
        uncertainty_plot,
        ax=axes[1],
        label="jackknife field standard deviation",
        shrink=0.68,
        pad=0.025,
        aspect=24,
    )
    axes[1].set_title("Accumulated Uncertainty")
    _plot_points(axes[1], X, removed_index, recommended_point)

    for axis in axes:
        axis.set(
            xlim=(-1.0, 1.0),
            ylim=(-1.0, 1.0),
            aspect="equal",
            xlabel="x",
            ylabel="y",
        )
    return _figure_to_image(figure)


def _plot_points(
    axis: plt.Axes,
    X: np.ndarray,
    removed_index: int | None,
    recommended_point: np.ndarray | None,
) -> None:
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
    if removed_index is not None:
        axis.scatter(
            X[removed_index, 0],
            X[removed_index, 1],
            c="red",
            marker="x",
            s=170,
            linewidths=2.8,
            label="omitted in current fold",
            zorder=5,
        )
    if recommended_point is not None:
        axis.scatter(
            recommended_point[0],
            recommended_point[1],
            c="gold",
            edgecolors="black",
            marker="*",
            s=220,
            linewidths=1.0,
            label="recommended next point",
            zorder=6,
            clip_on=False,
        )
    axis.legend(loc="upper left", fontsize=8)


def _recommended_point(grid: np.ndarray, X: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    """Choose the most uncertain grid location that is not already sampled."""

    normalized_grid = DOMAIN.normalize(grid)
    normalized_observations = DOMAIN.normalize(X)
    distances = np.linalg.norm(
        normalized_grid[:, None, :] - normalized_observations[None, :, :], axis=2
    )
    eligible = np.min(distances, axis=1) >= 0.08
    if not np.any(eligible):
        raise ValueError("The recommendation grid contains no unsampled locations.")
    scores = uncertainty.reshape(-1).copy()
    scores[~eligible] = -np.inf
    return grid[int(np.argmax(scores))].copy()


def _figure_to_image(figure: plt.Figure) -> Image.Image:
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    image = Image.fromarray(rgba.reshape(height, width, 4)).convert("P", palette=Image.Palette.ADAPTIVE)
    plt.close(figure)
    return image


if __name__ == "__main__":
    main()
