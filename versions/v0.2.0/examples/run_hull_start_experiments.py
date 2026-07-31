"""Run KRISP-U from four corner points plus one random interior point."""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray
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

DATASETS = (
    "quadratic_bowl_2d",
    "branin",
    "six_hump_camel",
    "ackley_2d",
    "gaussian_mixture_2d",
)
N_ITERATIONS = 24
ACQUISITION = "kld"


@dataclass
class HullStartTrace:
    """Trace for one hull-start KRISP-U run."""

    dataset: str
    iteration: int
    n_observed: int
    domain_rmse: float
    mean_domain_uncertainty: float
    max_domain_uncertainty: float
    mean_candidate_acquisition_score: float
    max_candidate_acquisition_score: float
    selected_acquisition_score: float
    selected_kernel_family: str | None
    selected_kernel_repr: str | None
    kernel_score_margin: float | None
    kernel_best_score: float | None
    kernel_second_best_score: float | None
    next_x1: float
    next_x2: float
    next_y: float


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    output_dir = Path("benchmark_outputs/hull_start")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, float | int | str | None]] = []
    final_rows: list[dict[str, float | int | str | None]] = []
    for offset, dataset_name in enumerate(DATASETS):
        dataset = get_dataset(dataset_name)
        trace, X_final, y_final = run_hull_start_experiment(
            dataset,
            n_iterations=N_ITERATIONS,
            grid_size=90,
            candidate_grid_size=45,
            random_state=151 + offset,
        )
        rows = [row.__dict__ for row in trace]
        all_rows.extend(rows)
        final_rows.append(_final_summary_row(dataset, trace, X_final, y_final))
        _write_csv(output_dir / f"{dataset.name}_hull_start_trace.csv", rows)
        _plot_trace(dataset, trace, output_dir)
        _plot_trajectory(dataset, X_final, output_dir)
        _plot_summary_3x1(dataset, trace, X_final, output_dir)

    _write_csv(output_dir / "hull_start_summary.csv", all_rows)
    _write_csv(output_dir / "hull_start_final_summary.csv", final_rows)
    print(f"Wrote hull-start experiment outputs to {output_dir}")


def run_hull_start_experiment(
    dataset: ToyDataset,
    n_iterations: int = 18,
    grid_size: int = 90,
    candidate_grid_size: int = 45,
    random_state: int = 151,
) -> tuple[list[HullStartTrace], NDArray[np.float64], NDArray[np.float64]]:
    """Run KRISP-U from corners plus one interior point."""

    if dataset.dimension != 2:
        raise ValueError("Hull-start experiment script expects 2D datasets.")
    if dataset.candidates is not None:
        raise ValueError("Hull-start experiment expects continuous toy datasets.")

    acquisition_name = normalize_acquisition_name(ACQUISITION)
    space = ContinuousSpace(dataset.bounds, names=dataset.labels)
    X = corner_plus_interior_design(dataset.bounds, random_state=random_state)
    y = dataset.evaluate(X)
    domain_points = _make_grid(dataset.bounds, grid_size)
    candidate_points = _make_grid(dataset.bounds, candidate_grid_size)
    true_values = dataset.evaluate(domain_points)
    rows: list[HullStartTrace] = []
    kernel_prior_config = KernelPriorConfig(random_state=random_state)
    gpr_config = GprConfig(
        n_restarts_optimizer=0,
        random_state=random_state,
        adaptive_kernel=True,
        kernel_prior_config=kernel_prior_config,
    )

    for iteration in range(n_iterations):
        optimizer = KrispUOptimizer(
            space,
            objective=dataset.objective,
            acquisition=acquisition_name,
            n_candidates=len(candidate_points),
            random_state=random_state + iteration,
            optimize_continuous_acquisition=False,
            gpr_config=gpr_config,
            kernel_prior_config=kernel_prior_config,
        )
        optimizer.fit(X, y)
        prediction, uncertainty = optimizer.predict(domain_points)
        _, _, candidate_scores = optimizer._score_candidates(
            candidate_points, acquisition_name
        )
        acquisition = optimizer.ask(candidates=candidate_points)
        next_y = float(dataset.evaluate(acquisition.x_next.reshape(1, -1))[0])
        kernel_prior_result = optimizer.kernel_prior_result_
        rows.append(
            HullStartTrace(
                dataset=dataset.name,
                iteration=iteration,
                n_observed=len(X),
                domain_rmse=float(np.sqrt(np.mean((prediction - true_values) ** 2))),
                mean_domain_uncertainty=float(np.mean(uncertainty)),
                max_domain_uncertainty=float(np.max(uncertainty)),
                mean_candidate_acquisition_score=float(np.mean(candidate_scores)),
                max_candidate_acquisition_score=float(np.max(candidate_scores)),
                selected_acquisition_score=float(acquisition.score),
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
                next_x1=float(acquisition.x_next[0]),
                next_x2=float(acquisition.x_next[1]),
                next_y=next_y,
            )
        )
        X = np.vstack((X, acquisition.x_next.reshape(1, -1)))
        y = np.append(y, next_y)

    return rows, X, y


def _make_grid(
    bounds: NDArray[np.float64], points_per_axis: int
) -> NDArray[np.float64]:
    x_axis = np.linspace(bounds[0, 0], bounds[0, 1], points_per_axis)
    y_axis = np.linspace(bounds[1, 0], bounds[1, 1], points_per_axis)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    return np.column_stack((grid_x.ravel(), grid_y.ravel()))


def _plot_trace(
    dataset: ToyDataset,
    trace: list[HullStartTrace],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    _draw_rmse_trace(axes[0], trace)
    _draw_uncertainty_trace(axes[1], trace)
    fig.suptitle(f"{dataset.name}: adaptive KLD hull-start KRISP-U")
    fig.savefig(output_dir / f"{dataset.name}_hull_start_trace.png", dpi=200)
    plt.close(fig)


def _final_summary_row(
    dataset: ToyDataset,
    trace: list[HullStartTrace],
    X_final: NDArray[np.float64],
    y_final: NDArray[np.float64],
) -> dict[str, float | int | str | None]:
    initial = trace[0]
    final = trace[-1]
    _ = dataset, y_final
    return {
        "dataset": trace[0].dataset,
        "n_initial": 5,
        "n_final": int(len(X_final)),
        "initial_domain_rmse": initial.domain_rmse,
        "final_domain_rmse": final.domain_rmse,
        "rmse_reduction_fraction": (
            (initial.domain_rmse - final.domain_rmse) / initial.domain_rmse
        ),
        "initial_mean_domain_uncertainty": initial.mean_domain_uncertainty,
        "final_mean_domain_uncertainty": final.mean_domain_uncertainty,
        "mean_uncertainty_reduction_fraction": (
            (initial.mean_domain_uncertainty - final.mean_domain_uncertainty)
            / initial.mean_domain_uncertainty
        ),
        "initial_max_domain_uncertainty": initial.max_domain_uncertainty,
        "final_max_domain_uncertainty": final.max_domain_uncertainty,
        "max_uncertainty_reduction_fraction": (
            (initial.max_domain_uncertainty - final.max_domain_uncertainty)
            / initial.max_domain_uncertainty
        ),
    }


def _plot_trajectory(
    dataset: ToyDataset,
    X_final: NDArray[np.float64],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    _draw_trajectory(ax, fig, dataset, X_final)
    fig.savefig(output_dir / f"{dataset.name}_hull_start_trajectory.png", dpi=200)
    plt.close(fig)


def _plot_summary_3x1(
    dataset: ToyDataset,
    trace: list[HullStartTrace],
    X_final: NDArray[np.float64],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.5, 12),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.0, 1.0]},
    )
    _draw_trajectory(axes[0], fig, dataset, X_final)
    _draw_rmse_trace(axes[1], trace)
    _draw_uncertainty_trace(axes[2], trace)
    fig.suptitle(
        f"{dataset.name}: adaptive KLD hull-start field reconstruction", fontsize=13
    )
    fig.savefig(output_dir / f"{dataset.name}_hull_start_summary_3x1.png", dpi=200)
    plt.close(fig)


def _draw_trajectory(
    ax: plt.Axes,
    fig: plt.Figure,
    dataset: ToyDataset,
    X_final: NDArray[np.float64],
) -> None:
    labels = list(dataset.labels or ("x1", "x2"))
    x_axis = np.linspace(dataset.bounds[0, 0], dataset.bounds[0, 1], 180)
    y_axis = np.linspace(dataset.bounds[1, 0], dataset.bounds[1, 1], 180)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    values = dataset.evaluate(np.column_stack((grid_x.ravel(), grid_y.ravel())))

    contour = ax.contourf(grid_x, grid_y, values.reshape(grid_x.shape), levels=45)
    fig.colorbar(contour, ax=ax, label="response")

    ax.scatter(
        X_final[:5, 0],
        X_final[:5, 1],
        color="white",
        edgecolor="black",
        s=60,
        label="initial hull + interior",
        zorder=4,
    )
    additions = X_final[5:]
    ax.plot(additions[:, 0], additions[:, 1], color="white", linewidth=1.2, alpha=0.85)
    ax.scatter(
        additions[:, 0],
        additions[:, 1],
        color="tab:red",
        edgecolor="black",
        s=46,
        label="adaptive KLD KRISP-U additions",
        zorder=5,
    )

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(f"{dataset.name}: hull-start field-sampling trajectory")
    ax.legend(loc="best")


def _draw_rmse_trace(ax: plt.Axes, trace: list[HullStartTrace]) -> None:
    iterations = [row.iteration for row in trace]
    rmse = [row.domain_rmse for row in trace]
    ax.plot(iterations, rmse, marker="o")
    ax.set_xlabel("iteration")
    ax.set_ylabel("domain RMSE")
    ax.set_title("Model error")


def _draw_uncertainty_trace(ax: plt.Axes, trace: list[HullStartTrace]) -> None:
    iterations = [row.iteration for row in trace]
    mean_scores = [row.mean_candidate_acquisition_score for row in trace]
    max_scores = [row.max_candidate_acquisition_score for row in trace]
    ax.plot(
        iterations,
        mean_scores,
        marker="o",
        color="tab:green",
        label="mean",
    )
    ax.plot(
        iterations,
        max_scores,
        marker="s",
        color="tab:orange",
        label="max",
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel("KLD information gain")
    ax.set_title("Candidate acquisition score")
    ax.legend(loc="best")


def _write_csv(path: Path, rows: list[dict[str, float | int | str | None]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
