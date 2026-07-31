"""Generate side-by-side KRISP-U prediction and uncertainty GIFs."""

from __future__ import annotations

import argparse
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from sklearn.exceptions import ConvergenceWarning

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from krispu import (  # noqa: E402
    KernelPriorConfig,
    KrispUOptimizer,
    corner_plus_interior_design,
    get_dataset,
)
from krispu.acquisition import normalize_acquisition_name  # noqa: E402
from krispu.datasets import ToyDataset  # noqa: E402
from krispu.models import GprConfig  # noqa: E402
from krispu.space import ContinuousSpace  # noqa: E402

DEFAULT_DATASETS = ("quadratic_bowl_2d", "branin", "gaussian_mixture_2d")


@dataclass
class GifFrame:
    """State needed to draw one GIF frame."""

    iteration: int
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    prediction: NDArray[np.float64]
    predictive_std: NDArray[np.float64]
    acquisition_score: NDArray[np.float64]
    acquisition: str
    next_point: NDArray[np.float64]
    rmse: float
    selected_kernel_family: str | None
    selected_kernel_repr: str | None
    kernel_score_margin: float | None
    kernel_best_score: float | None
    kernel_second_best_score: float | None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create KRISP-U prediction/uncertainty GIFs for 2D toy datasets."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--output-dir", default="benchmark_outputs/gifs")
    parser.add_argument("--initial-n", type=int, default=5)
    parser.add_argument(
        "--initial-design",
        choices=("lhs", "hull"),
        default="lhs",
        help="Initial design: lhs or hull for corners plus one interior point.",
    )
    parser.add_argument("--max-iterations", type=int, default=18)
    parser.add_argument("--grid-size", type=int, default=65)
    parser.add_argument("--candidate-grid-size", type=int, default=45)
    parser.add_argument("--random-state", type=int, default=31)
    parser.add_argument("--duration-ms", type=int, default=900)
    parser.add_argument(
        "--acquisition",
        default="kld",
        help="Acquisition to visualize and use for next-point selection.",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Keep every KRISP-U step instead of only frames with improved domain RMSE.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for offset, dataset_name in enumerate(args.datasets):
        dataset = get_dataset(dataset_name)
        if args.initial_design == "hull":
            output_path = (
                output_dir / f"{dataset.name}_hull_start_prediction_uncertainty.gif"
            )
        else:
            output_path = output_dir / f"{dataset.name}_prediction_uncertainty.gif"
        generate_prediction_uncertainty_gif(
            dataset,
            output_path=output_path,
            initial_n=args.initial_n,
            initial_design=args.initial_design,
            max_iterations=args.max_iterations,
            grid_size=args.grid_size,
            candidate_grid_size=args.candidate_grid_size,
            random_state=args.random_state + offset,
            duration_ms=args.duration_ms,
            acquisition=args.acquisition,
            improvement_only=not args.all_frames,
        )
        print(f"Wrote {output_path}")


def generate_prediction_uncertainty_gif(
    dataset: ToyDataset,
    output_path: Path,
    initial_n: int = 5,
    initial_design: str = "lhs",
    max_iterations: int = 18,
    grid_size: int = 65,
    candidate_grid_size: int = 45,
    random_state: int = 31,
    duration_ms: int = 900,
    acquisition: str = "kld",
    improvement_only: bool = True,
) -> None:
    """Generate a side-by-side prediction and uncertainty animation."""

    if dataset.dimension != 2:
        raise ValueError("GIF generation currently requires a 2D toy dataset.")
    if initial_n < 2:
        raise ValueError("initial_n must be at least 2.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    space = ContinuousSpace(dataset.bounds, names=dataset.labels)
    grid_x, grid_y, domain_points = _make_grid(dataset.bounds, grid_size)
    _, _, candidate_points = _make_grid(dataset.bounds, candidate_grid_size)
    true_values = dataset.evaluate(domain_points).reshape(grid_x.shape)
    y_min = float(np.nanmin(true_values))
    y_max = float(np.nanmax(true_values))

    X = _initial_design(
        dataset=dataset,
        initial_n=initial_n,
        initial_design=initial_design,
        random_state=random_state,
    )
    y = dataset.evaluate(X)
    frames = _collect_frames(
        dataset=dataset,
        space=space,
        X=X,
        y=y,
        domain_points=domain_points,
        true_values=true_values.ravel(),
        grid_shape=grid_x.shape,
        candidate_points=candidate_points,
        max_iterations=max_iterations,
        random_state=random_state,
        acquisition=normalize_acquisition_name(acquisition),
    )
    selected_frames = _select_frames(frames, improvement_only=improvement_only)
    _write_frame_trace(
        output_path.with_suffix(".csv"),
        selected_frames,
    )
    images = [
        _draw_frame(
            dataset=dataset,
            frame=frame,
            grid_x=grid_x,
            grid_y=grid_y,
            y_min=y_min,
            y_max=y_max,
            uncertainty_max=max(_max_uncertainty(selected_frames), 1e-12),
            acquisition_max=max(_max_acquisition_score(selected_frames), 1e-12),
        )
        for frame in selected_frames
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def _initial_design(
    dataset: ToyDataset,
    initial_n: int,
    initial_design: str,
    random_state: int,
) -> NDArray[np.float64]:
    if initial_design == "lhs":
        return dataset.initial_design(
            initial_n, method="lhs", random_state=random_state
        )
    if initial_design == "hull":
        return corner_plus_interior_design(dataset.bounds, random_state=random_state)
    raise ValueError("initial_design must be either 'lhs' or 'hull'.")


def _collect_frames(
    dataset: ToyDataset,
    space: ContinuousSpace,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    domain_points: NDArray[np.float64],
    true_values: NDArray[np.float64],
    grid_shape: tuple[int, int],
    candidate_points: NDArray[np.float64],
    max_iterations: int,
    random_state: int,
    acquisition: str,
) -> list[GifFrame]:
    frames: list[GifFrame] = []
    acquisition_method = normalize_acquisition_name(acquisition)
    kernel_prior_config = KernelPriorConfig(random_state=random_state)
    gpr_config = GprConfig(
        n_restarts_optimizer=0,
        random_state=random_state,
        adaptive_kernel=True,
        kernel_prior_config=kernel_prior_config,
    )
    for iteration in range(max_iterations):
        optimizer = KrispUOptimizer(
            space,
            objective=dataset.objective,
            acquisition=acquisition_method,
            gpr_config=gpr_config,
            kernel_prior_config=kernel_prior_config,
            random_state=random_state + iteration,
            optimize_continuous_acquisition=False,
        )
        optimizer.fit(X, y)
        prediction, predictive_std = optimizer.predict(domain_points)
        _, _, domain_scores = optimizer._score_candidates(
            domain_points, acquisition_method
        )
        acquisition_result = optimizer.ask(candidates=candidate_points)
        rmse = float(np.sqrt(np.mean((prediction - true_values) ** 2)))
        kernel_prior_result = optimizer.kernel_prior_result_
        frames.append(
            GifFrame(
                iteration=iteration,
                X=X.copy(),
                y=y.copy(),
                prediction=prediction.reshape(grid_shape),
                predictive_std=predictive_std.reshape(grid_shape),
                acquisition_score=domain_scores.reshape(grid_shape),
                acquisition=acquisition_result.acquisition,
                next_point=acquisition_result.x_next.copy(),
                rmse=rmse,
                selected_kernel_family=(
                    None
                    if kernel_prior_result is None
                    else kernel_prior_result.selected_family
                ),
                selected_kernel_repr=(
                    None
                    if kernel_prior_result is None
                    else kernel_prior_result.selected_kernel_repr
                ),
                kernel_score_margin=(
                    None
                    if kernel_prior_result is None
                    else kernel_prior_result.score_margin
                ),
                kernel_best_score=(
                    None
                    if kernel_prior_result is None
                    else kernel_prior_result.best_score
                ),
                kernel_second_best_score=(
                    None
                    if kernel_prior_result is None
                    else kernel_prior_result.second_best_score
                ),
            )
        )
        next_y = dataset.evaluate(acquisition_result.x_next.reshape(1, -1))
        X = np.vstack((X, acquisition_result.x_next.reshape(1, -1)))
        y = np.append(y, next_y)
    return frames


def _select_frames(
    frames: list[GifFrame], improvement_only: bool = True
) -> list[GifFrame]:
    if not improvement_only:
        return frames

    selected: list[GifFrame] = []
    best_rmse = float("inf")
    for frame in frames:
        if frame.rmse <= best_rmse:
            selected.append(frame)
            best_rmse = frame.rmse

    if len(selected) >= 4:
        return selected

    # If a difficult surface gives only a few strict improvements, keep every
    # frame but the title will still show the actual domain RMSE.
    return frames


def _draw_frame(
    dataset: ToyDataset,
    frame: GifFrame,
    grid_x: NDArray[np.float64],
    grid_y: NDArray[np.float64],
    y_min: float,
    y_max: float,
    uncertainty_max: float,
    acquisition_max: float,
) -> Image.Image:
    labels = list(dataset.labels or ("x1", "x2"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    fig.suptitle(
        (
            f"{dataset.name} | iteration {frame.iteration} | "
            f"measured {len(frame.X)} | domain RMSE {frame.rmse:.4g} | "
            f"kernel {frame.selected_kernel_family or 'fixed'} | "
            f"acq {frame.acquisition}"
        ),
        fontsize=12,
    )

    pred = axes[0].contourf(
        grid_x,
        grid_y,
        frame.prediction,
        levels=40,
        cmap="viridis",
        vmin=y_min,
        vmax=y_max,
    )
    fig.colorbar(pred, ax=axes[0], label="predicted response")
    _overlay_points(axes[0], frame, labels)
    axes[0].set_title("GPR prediction of domain")

    uncert = axes[1].contourf(
        grid_x,
        grid_y,
        frame.acquisition_score,
        levels=40,
        cmap="magma",
        vmin=0.0,
        vmax=acquisition_max,
    )
    fig.colorbar(uncert, ax=axes[1], label=_acquisition_colorbar_label(frame))
    _overlay_points(axes[1], frame, labels)
    axes[1].set_title("Acquisition score and next selected point")

    for ax in axes:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_xlim(float(np.min(grid_x)), float(np.max(grid_x)))
        ax.set_ylim(float(np.min(grid_y)), float(np.max(grid_y)))

    image = _figure_to_image(fig)
    plt.close(fig)
    return image


def _overlay_points(ax: plt.Axes, frame: GifFrame, labels: list[str]) -> None:
    _ = labels
    ax.scatter(
        frame.X[:, 0],
        frame.X[:, 1],
        c="white",
        edgecolor="black",
        s=34,
        linewidth=0.8,
        label="measured",
        zorder=4,
    )
    ax.scatter(
        frame.next_point[0],
        frame.next_point[1],
        c="cyan",
        edgecolor="black",
        marker="*",
        s=180,
        label="next",
        zorder=5,
    )
    ax.legend(loc="upper right", fontsize=8)


def _figure_to_image(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba.reshape(height, width, 4)).convert(
        "P", palette=Image.Palette.ADAPTIVE
    )


def _make_grid(
    bounds: NDArray[np.float64], grid_size: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")
    x_axis = np.linspace(bounds[0, 0], bounds[0, 1], grid_size)
    y_axis = np.linspace(bounds[1, 0], bounds[1, 1], grid_size)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    return grid_x, grid_y, points


def _max_uncertainty(frames: list[GifFrame]) -> float:
    return float(max(np.nanmax(frame.predictive_std) for frame in frames))


def _max_acquisition_score(frames: list[GifFrame]) -> float:
    return float(max(np.nanmax(frame.acquisition_score) for frame in frames))


def _acquisition_colorbar_label(frame: GifFrame) -> str:
    if frame.acquisition == "kld":
        return "expected KL information gain"
    if frame.acquisition == "uncertainty":
        return "predictive std"
    return "acquisition score"


def _write_frame_trace(
    output_path: Path,
    frames: list[GifFrame],
) -> None:
    rows = []
    for frame in frames:
        rows.append(
            {
                "iteration": frame.iteration,
                "n_measured": int(len(frame.X)),
                "domain_rmse": frame.rmse,
                "selected_kernel_family": frame.selected_kernel_family,
                "selected_kernel_repr": frame.selected_kernel_repr,
                "kernel_score_margin": frame.kernel_score_margin,
                "kernel_best_score": frame.kernel_best_score,
                "kernel_second_best_score": frame.kernel_second_best_score,
                "acquisition": frame.acquisition,
                "mean_domain_uncertainty": float(np.mean(frame.predictive_std)),
                "max_domain_uncertainty": float(np.max(frame.predictive_std)),
                "mean_domain_acquisition_score": float(
                    np.mean(frame.acquisition_score)
                ),
                "max_domain_acquisition_score": float(np.max(frame.acquisition_score)),
                "next_x1": float(frame.next_point[0]),
                "next_x2": float(frame.next_point[1]),
            }
        )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
