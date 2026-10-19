"""Render tutorial GIFs with the v0.1.0 KRISP-U implementation."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as PolygonPath
from PIL import Image
from scipy.interpolate import RegularGridInterpolator


HERE = Path(__file__).resolve().parent
LEGACY_SOURCE = HERE.parents[2] / "versions" / "v0.1.0" / "source"
sys.path.insert(0, str(LEGACY_SOURCE))

from KRISPU import KRISPU
from Utilities import KLD
from pykrige.uk import UniversalKriging


FRAME_DURATION_MS = 1_000
N_ITERATIONS = 9
PREDICTION_LIMITS = (0.0, 1.0)
PREDICTION_TICKS = np.linspace(0.0, 1.0, 6)
UNCERTAINTY_TICKS = np.linspace(0.0, 1.0, 6)
GRID = np.linspace(4.0, 11.0, 200)
HULL = np.array([[5.0, 5.0], [10.0, 5.0], [10.0, 10.0], [5.0, 10.0]])
INITIAL_POINTS = np.array(
    [
        [5.0, 5.0], [10.0, 5.0], [10.0, 10.0], [5.0, 10.0],
        [5.5, 7.5], [7.5, 6.5], [8.5, 8.5], [9.25, 7.5],
    ]
)
SAMPLE_COORDINATES = np.arange(5.25, 10.0, 0.25)
TOY_LABELS = ("Value A", "Value B", "Value C")
ORIGINAL_LABELS = ("Peak", "width", "# to failure (output)")


def main() -> None:
    prediction_frames: list[Image.Image] = []
    uncertainty_frames: list[Image.Image] = []
    round_robin_frames: list[Image.Image] = []
    original_prediction_frames: list[Image.Image] = []
    original_uncertainty_frames: list[Image.Image] = []
    original_round_robin_frames: list[Image.Image] = []
    sum_uncertainties: list[float] = []
    points = INITIAL_POINTS.copy()
    values = _original_like_field(points[:, 0], points[:, 1])

    for iteration in range(N_ITERATIONS):
        prediction, uncertainty, inside_hull, sum_uncertainty = _surfaces(points, values)
        sum_uncertainties.append(sum_uncertainty)
        next_point = _highest_uncertainty_grid_point(uncertainty, points)
        prediction_frames.append(
            _single_panel(
                prediction,
                inside_hull,
                points,
                next_point,
                iteration,
                cmap="viridis",
                limits=PREDICTION_LIMITS,
                colorbar_label=TOY_LABELS[2],
                title="Prediction",
                axis_labels=TOY_LABELS[:2],
            )
        )
        uncertainty_frames.append(
            _single_panel(
                uncertainty,
                inside_hull,
                points,
                next_point,
                iteration,
                cmap="magma",
                limits=(0.0, 1.0),
                colorbar_label="Uncertainty",
                title="Uncertainty",
                axis_labels=TOY_LABELS[:2],
            )
        )
        round_robin_frames.extend([
            _single_panel(prediction, inside_hull, points, None, iteration, cmap="viridis", limits=PREDICTION_LIMITS, colorbar_label=TOY_LABELS[2], title="Prediction", axis_labels=TOY_LABELS[:2]),
            _single_panel(uncertainty, inside_hull, points, None, iteration, cmap="magma", limits=(0.0, 1.0), colorbar_label="Uncertainty", title="Uncertainty", axis_labels=TOY_LABELS[:2]),
            _single_panel(uncertainty, inside_hull, points, next_point, iteration, cmap="magma", limits=(0.0, 1.0), colorbar_label="Uncertainty", title="Selected Max Uncertainty", axis_labels=TOY_LABELS[:2]),
        ])
        original_prediction_frames.append(
            _single_panel(
                prediction, inside_hull, points, next_point, iteration,
                cmap="viridis", limits=PREDICTION_LIMITS,
                colorbar_label=ORIGINAL_LABELS[2], title="Prediction",
                axis_labels=ORIGINAL_LABELS[:2],
            )
        )
        original_uncertainty_frames.append(
            _single_panel(
                uncertainty, inside_hull, points, next_point, iteration,
                cmap="magma", limits=(0.0, 1.0), colorbar_label="Uncertainty",
                title="Uncertainty", axis_labels=ORIGINAL_LABELS[:2],
            )
        )
        next_value = _original_like_field(next_point[0], next_point[1])
        points = np.vstack((points, next_point))
        values = np.append(values, next_value)
        updated_prediction, _, updated_hull, _ = _surfaces(points, values)
        round_robin_frames.append(
            _single_panel(updated_prediction, updated_hull, points, None, iteration + 1, cmap="viridis", limits=PREDICTION_LIMITS, colorbar_label=TOY_LABELS[2], title="Updated Prediction", axis_labels=TOY_LABELS[:2])
        )
        original_round_robin_frames.extend([
            _single_panel(prediction, inside_hull, points[:-1], None, iteration, cmap="viridis", limits=PREDICTION_LIMITS, colorbar_label=ORIGINAL_LABELS[2], title="Prediction", axis_labels=ORIGINAL_LABELS[:2]),
            _single_panel(uncertainty, inside_hull, points[:-1], None, iteration, cmap="magma", limits=(0.0, 1.0), colorbar_label="Uncertainty", title="Uncertainty", axis_labels=ORIGINAL_LABELS[:2]),
            _single_panel(uncertainty, inside_hull, points[:-1], next_point, iteration, cmap="magma", limits=(0.0, 1.0), colorbar_label="Uncertainty", title="Selected Max Uncertainty", axis_labels=ORIGINAL_LABELS[:2]),
            _single_panel(updated_prediction, updated_hull, points, None, iteration + 1, cmap="viridis", limits=PREDICTION_LIMITS, colorbar_label=ORIGINAL_LABELS[2], title="Updated Prediction", axis_labels=ORIGINAL_LABELS[:2]),
        ])

    _save_gif(HERE / "Gif.gif", uncertainty_frames)
    _save_gif(HERE / "Prediction.gif", prediction_frames)
    _save_gif(HERE / "Prediction_Uncertainty.gif", round_robin_frames)
    _save_gif(HERE / "Gif_original_labels.gif", original_uncertainty_frames)
    _save_gif(HERE / "Prediction_original_labels.gif", original_prediction_frames)
    _save_gif(HERE / "Prediction_Uncertainty_original_labels.gif", original_round_robin_frames)
    _plot_sum_uncertainty(HERE / "gif_sum_uncertainties.png", sum_uncertainties)


def _surfaces(
    points: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit and evaluate the unmodified v0.1.0 KRISP-U model on the tutorial grid."""

    grid_x, grid_y = np.meshgrid(GRID, GRID)
    query = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    krispu = KRISPU(
        points.astype(float),
        values.astype(float),
        model_class=UniversalKriging,
        model_kwargs={"variogram_model": "spherical", "verbose": False, "enable_plotting": False},
    )
    prediction = np.asarray(krispu.fit(GRID, GRID), dtype=float)
    sum_uncertainty = krispu.evaluate(KLD)
    uncertainty = np.asarray(krispu.generate_uncertainty_map(GRID, GRID, method="cubic"), dtype=float)
    inside_hull = PolygonPath(HULL).contains_points(query).reshape(grid_x.shape)
    prediction[~inside_hull] = np.nan
    uncertainty[~inside_hull] = np.nan
    uncertainty[uncertainty == 0.0] = np.nan
    return prediction, uncertainty, inside_hull, sum_uncertainty


def _plot_sum_uncertainty(path: Path, sum_uncertainties: list[float]) -> None:
    """Plot the current GIF case without changing the original tutorial figure."""

    figure, axis = plt.subplots(figsize=(10, 4.5), layout="constrained")
    iterations = np.arange(1, len(sum_uncertainties) + 1)
    axis.plot(iterations, sum_uncertainties, marker="o")
    axis.set(
        xlabel="Points added",
        ylabel="Total uncertainty",
        xlim=(0.5, len(sum_uncertainties) + 0.5),
        ylim=(0.0, 0.30),
    )
    axis.set_xticks([1, 3, 5, 7, 9])
    axis.set_yticks(np.arange(0.0, 0.31, 0.05))
    axis.grid()
    figure.savefig(path, dpi=300)
    plt.close(figure)
    print(f"Wrote {path}")


def _highest_uncertainty_grid_point(uncertainty: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Choose the most uncertain untested location on the fixed 0.25 sampling grid."""

    sample_x, sample_y = np.meshgrid(SAMPLE_COORDINATES, SAMPLE_COORDINATES)
    candidates = np.column_stack((sample_x.ravel(), sample_y.ravel()))
    interpolate_uncertainty = RegularGridInterpolator(
        (GRID, GRID), uncertainty, bounds_error=False, fill_value=np.nan
    )
    # KRISP-U returns a grid indexed as (y, x), whereas plot coordinates are (x, y).
    candidate_uncertainty = interpolate_uncertainty(candidates[:, ::-1])
    already_sampled = np.any(
        np.all(np.isclose(candidates[:, None, :], points[None, :, :]), axis=2), axis=1
    )
    candidate_uncertainty[already_sampled] = np.nan
    return candidates[np.nanargmax(candidate_uncertainty)]


def _original_like_field(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
    """A normalized broad trend with several smooth peaks and troughs."""

    vertical_rise = (10.0 - y) / 5.0
    lateral_rise = (x - 5.0) / 5.0
    broad_trend = 0.08 + 0.62 * vertical_rise + 0.16 * lateral_rise
    lower_peak = 0.20 * np.exp(-((x - 7.4) ** 2 / 1.5 + (y - 6.4) ** 2 / 0.8))
    upper_trough = -0.18 * np.exp(-((x - 8.4) ** 2 / 0.9 + (y - 8.6) ** 2 / 1.2))
    side_peak = 0.12 * np.exp(-((x - 9.2) ** 2 / 0.5 + (y - 7.5) ** 2 / 1.8))
    curved_ridge = 0.10 * np.exp(-((y - (5.8 + 0.35 * np.sin(1.5 * (x - 6.0)))) ** 2 / 0.18))
    western_trough = -0.09 * np.exp(-((x - 6.1) ** 2 / 0.7 + (y - 7.7) ** 2 / 0.55))
    return np.clip(
        broad_trend + lower_peak + upper_trough + side_peak + curved_ridge + western_trough,
        0.0,
        1.0,
    )


def _single_panel(
    surface: np.ndarray,
    inside_hull: np.ndarray,
    points: np.ndarray,
    next_point: np.ndarray | None,
    iteration: int,
    *,
    cmap: str,
    limits: tuple[float, float],
    colorbar_label: str,
    title: str,
    axis_labels: tuple[str, str],
) -> Image.Image:
    figure, axis = plt.subplots(figsize=(10, 7.5), layout="constrained")
    image = axis.imshow(
        np.ma.masked_where(~inside_hull, surface),
        extent=(GRID.min(), GRID.max(), GRID.min(), GRID.max()),
        origin="lower",
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
    )
    _decorate_axis(axis, points, next_point, f"{title} Iteration {iteration + 1}", axis_labels)
    colorbar = figure.colorbar(image, ax=axis, label=colorbar_label)
    colorbar.set_ticks(PREDICTION_TICKS if limits == PREDICTION_LIMITS else UNCERTAINTY_TICKS)
    colorbar.ax.tick_params(labelsize=18)
    colorbar.set_label(colorbar_label, fontsize=20)
    return _figure_image(figure)


def _joint_panel(
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    inside_hull: np.ndarray,
    points: np.ndarray,
    next_point: np.ndarray | None,
    iteration: int,
    *,
    axis_labels: tuple[str, str, str],
) -> Image.Image:
    figure, axes = plt.subplots(1, 2, figsize=(16, 7), layout="constrained", sharex=True, sharey=True)
    panels = (
        (axes[0], uncertainty, "magma", (0.0, 1.0), "Uncertainty", "Uncertainty"),
        (axes[1], prediction, "viridis", PREDICTION_LIMITS, axis_labels[2], "Prediction"),
    )
    for axis, surface, cmap, limits, label, title in panels:
        image = axis.imshow(
            np.ma.masked_where(~inside_hull, surface),
            extent=(GRID.min(), GRID.max(), GRID.min(), GRID.max()),
            origin="lower",
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
        )
        _decorate_axis(axis, points, next_point, f"{title} Iteration {iteration + 1}", axis_labels[:2])
        colorbar = figure.colorbar(image, ax=axis, label=label)
        colorbar.set_ticks(PREDICTION_TICKS if limits == PREDICTION_LIMITS else UNCERTAINTY_TICKS)
        colorbar.ax.tick_params(labelsize=16)
        colorbar.set_label(label, fontsize=18)
    return _figure_image(figure)


def _decorate_axis(
    axis: plt.Axes,
    points: np.ndarray,
    next_point: np.ndarray | None,
    title: str,
    axis_labels: tuple[str, str],
) -> None:
    axis.scatter(points[:, 0], points[:, 1], c="red", label="Data Points", s=50, zorder=3)
    if next_point is not None:
        axis.scatter(
            next_point[0],
            next_point[1],
            c="blue",
            label="Max Uncertainty",
            s=100,
            edgecolor="black",
            zorder=4,
        )
    else:
        axis.scatter(
            [],
            [],
            c="blue",
            label="Max Uncertainty",
            s=100,
            edgecolor="black",
        )
    axis.set(
        xlim=(4.0, 11.0),
        ylim=(4.0, 11.0),
        xlabel=axis_labels[0],
        ylabel=axis_labels[1],
        title=title,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(labelsize=18)
    axis.xaxis.label.set_size(20)
    axis.yaxis.label.set_size(20)
    axis.title.set_size(22)
    axis.legend(loc="lower center", fontsize=16)


def _figure_image(figure: plt.Figure) -> Image.Image:
    figure.canvas.draw()
    image = Image.fromarray(np.asarray(figure.canvas.buffer_rgba()), mode="RGBA").convert("RGB")
    plt.close(figure)
    return image


def _save_gif(path: Path, frames: list[Image.Image]) -> None:
    palette = _fixed_gif_palette()
    fixed_palette_frames = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in frames
    ]
    fixed_palette_frames[0].save(
        path,
        save_all=True,
        append_images=fixed_palette_frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=False,
    )
    print(f"Wrote {path}")


def _fixed_gif_palette() -> Image.Image:
    """Use one palette for every GIF frame so colors cannot flicker or shift."""

    cmap_colors = np.vstack([
        plt.get_cmap("viridis")(np.linspace(0.0, 1.0, 96))[:, :3],
        plt.get_cmap("magma")(np.linspace(0.0, 1.0, 96))[:, :3],
    ])
    grayscale = np.linspace(0.0, 1.0, 48)[:, None]
    grayscale = np.repeat(grayscale, 3, axis=1)
    accents = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    colors = np.rint(255.0 * np.vstack((cmap_colors, grayscale, accents))).astype(np.uint8)
    palette_values = colors.ravel().tolist()
    palette_values.extend([255, 255, 255] * (256 - len(colors)))
    palette = Image.new("P", (1, 1))
    palette.putpalette(palette_values)
    return palette


if __name__ == "__main__":
    main()
