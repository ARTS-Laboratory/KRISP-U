"""Generate KRISP-U field-reconstruction benchmark figures and tables.

This script is intentionally small enough to run on a laptop while still giving
evidence against random search and naive grid search on several 2D fields.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.exceptions import ConvergenceWarning

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from krispu import (
    BenchmarkResult,
    KernelPriorConfig,
    MethodTrace,
    get_dataset,
    run_benchmark,
)  # noqa: E402
from krispu.plotting import plot_benchmark_comparison  # noqa: E402
from krispu.space import DiscreteCandidateSpace  # noqa: E402

DATASETS = (
    "branin",
    "six_hump_camel",
    "ackley_2d",
    "quadratic_bowl_2d",
    "anisotropic_ridge",
    "gaussian_mixture_2d",
)
METHODS = ("krispu_fixed", "krispu_adaptive", "random", "grid", "lhs")
R2_THRESHOLD = 0.90
N_TRIALS = 3
DOMAIN_BUDGETS = {
    "branin": 64,
    "six_hump_camel": 48,
    "ackley_2d": 96,
    "quadratic_bowl_2d": 16,
    "anisotropic_ridge": 32,
    "gaussian_mixture_2d": 96,
}
BASE_BUDGET_SWEEP = (
    5,
    6,
    8,
    10,
    12,
    16,
    20,
    24,
    32,
    40,
    48,
    64,
    80,
    96,
)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main() -> None:
    output_dir = Path("benchmark_outputs")
    output_dir.mkdir(exist_ok=True)
    _remove_stale_threshold_plots(output_dir)

    summaries: list[dict[str, float | str]] = []
    comparisons: list[dict[str, float | str]] = []
    r2_comparisons: list[dict[str, float | str]] = []
    n_sweep: list[dict[str, float | str]] = []
    r2_threshold_rows: list[dict[str, float | str]] = []

    for dataset_name in DATASETS:
        domain_budget = _domain_budget(dataset_name)
        budget_sweep = _domain_budget_sweep(dataset_name)
        result = run_benchmark(
            dataset_name,
            methods=METHODS,
            budget=domain_budget,
            n_initial=None,
            n_trials=N_TRIALS,
            random_state=101,
            n_candidates=512,
            tolerance=_default_tolerance(dataset_name),
            optimize_continuous_acquisition=False,
            initial_design="hull",
            learning_curve_n_values=budget_sweep,
            adaptive_kernel=True,
            kernel_prior_config=KernelPriorConfig(random_state=101),
        )

        _save_benchmark_curve(result, output_dir)
        _save_surface_overlay(result, output_dir)
        _save_result_json(result, output_dir)

        for method, values in result.summary(n_bootstrap=500).items():
            summaries.append({"dataset": dataset_name, "method": method, **values})
        n_sweep.extend(_learning_summary_rows(result, budget_sweep, n_bootstrap=300))
        r2_threshold_rows.extend(
            _r2_threshold_summary_rows(
                result,
                threshold=R2_THRESHOLD,
                n_bootstrap=300,
            )
        )

        for baseline in ("random", "grid", "lhs"):
            comparison = result.compare_to_baseline(
                "krispu_adaptive",
                baseline,
                metric="field_nrmse",
                n_bootstrap=500,
            )
            comparisons.append(
                {
                    "dataset": dataset_name,
                    "baseline": baseline,
                    "metric": "field_nrmse",
                    **comparison,
                }
            )
            r2_comparison = result.compare_to_baseline(
                "krispu_adaptive",
                baseline,
                metric="field_r2",
                n_bootstrap=500,
            )
            r2_comparisons.append(
                {
                    "dataset": dataset_name,
                    "baseline": baseline,
                    "metric": "field_r2",
                    **r2_comparison,
                }
            )
        fixed_comparison = result.compare_to_baseline(
            "krispu_adaptive",
            "krispu_fixed",
            metric="field_r2",
            n_bootstrap=500,
        )
        r2_comparisons.append(
            {
                "dataset": dataset_name,
                "baseline": "krispu_fixed",
                "metric": "field_r2",
                **fixed_comparison,
            }
        )

    _write_csv(output_dir / "benchmark_summary.csv", summaries)
    _write_csv(
        output_dir / "krispu_normalized_field_error_comparisons.csv", comparisons
    )
    _write_csv(output_dir / "krispu_field_error_comparisons.csv", comparisons)
    _write_csv(output_dir / "krispu_r2_field_comparisons.csv", r2_comparisons)
    _save_normalized_field_error_summary(summaries, output_dir)
    _save_r2_field_score_summary(summaries, output_dir)
    _save_domain_panel_plot(output_dir)

    _write_csv(output_dir / "benchmark_n_sweep_summary.csv", n_sweep)
    _write_csv(output_dir / "r2_threshold_summary.csv", r2_threshold_rows)
    _save_n_sweep_plot(n_sweep, output_dir)
    _save_r2_sweep_plot(n_sweep, output_dir)
    _save_r2_threshold_plot(r2_threshold_rows, output_dir)
    _save_diagnostic_sweep_plot(
        n_sweep,
        output_dir,
        metric="field_p95_abs_error",
        ylabel="p95 absolute error",
        title="Tail reconstruction error vs measured points",
        filename="field_p95_abs_error_vs_n.png",
    )
    _save_diagnostic_sweep_plot(
        n_sweep,
        output_dir,
        metric="field_max_abs_error",
        ylabel="worst absolute error",
        title="Worst-region reconstruction error vs measured points",
        filename="field_max_abs_error_vs_n.png",
    )
    _save_diagnostic_sweep_plot(
        n_sweep,
        output_dir,
        metric="field_coverage_95",
        ylabel="fraction inside 95% GPR interval",
        title="Uncertainty calibration vs measured points",
        filename="field_coverage_95_vs_n.png",
    )
    _save_diagnostic_sweep_plot(
        n_sweep,
        output_dir,
        metric="mean_uncertainty",
        ylabel="mean predictive std",
        title="Integrated uncertainty vs measured points",
        filename="mean_uncertainty_vs_n.png",
    )


def _default_tolerance(dataset_name: str) -> float:
    if "quadratic" in dataset_name:
        return 0.10
    if "ackley" in dataset_name:
        return 0.75
    if "camel" in dataset_name:
        return 0.20
    if "branin" in dataset_name:
        return 1.0
    return 0.25


def _domain_budget(dataset_name: str) -> int:
    return DOMAIN_BUDGETS.get(dataset_name, max(BASE_BUDGET_SWEEP))


def _domain_budget_sweep(dataset_name: str) -> tuple[int, ...]:
    budget = _domain_budget(dataset_name)
    values = [value for value in BASE_BUDGET_SWEEP if value <= budget]
    if budget not in values:
        values.append(budget)
    return tuple(values)


def _save_benchmark_curve(result: BenchmarkResult, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_benchmark_comparison(result, ax=ax)
    ax.set_title(f"{result.dataset_name}: sampled response context")
    fig.tight_layout()
    fig.savefig(output_dir / f"{result.dataset_name}_benchmark_curves.png", dpi=200)
    plt.close(fig)


def _save_surface_overlay(result: BenchmarkResult, output_dir: Path) -> None:
    dataset = get_dataset(result.dataset_name)
    space = dataset.space()
    fig, ax = plt.subplots(figsize=(7, 6))

    if isinstance(space, DiscreteCandidateSpace):
        candidates = space.candidates
        values = dataset.evaluate(candidates)
        scatter = ax.scatter(
            candidates[:, 0],
            candidates[:, 1],
            c=values,
            cmap="viridis",
            s=36,
            alpha=0.80,
        )
        fig.colorbar(scatter, ax=ax, label="response")
    else:
        x_axis = np.linspace(dataset.bounds[0, 0], dataset.bounds[0, 1], 180)
        y_axis = np.linspace(dataset.bounds[1, 0], dataset.bounds[1, 1], 180)
        grid_x, grid_y = np.meshgrid(x_axis, y_axis)
        points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        values = dataset.evaluate(points).reshape(grid_x.shape)
        contour = ax.contourf(grid_x, grid_y, values, levels=45, cmap="viridis")
        fig.colorbar(contour, ax=ax, label="response")

    trace = _lowest_error_trace(result, _primary_krispu_method(result))
    ax.plot(trace.X[:, 0], trace.X[:, 1], color="white", linewidth=1.5, alpha=0.9)
    ax.scatter(
        trace.X[: result.n_initial, 0],
        trace.X[: result.n_initial, 1],
        color="white",
        edgecolor="black",
        s=55,
        label="initial",
        zorder=4,
    )
    ax.scatter(
        trace.X[result.n_initial :, 0],
        trace.X[result.n_initial :, 1],
        color="tab:red",
        edgecolor="black",
        s=55,
        label="KRISP-U additions",
        zorder=4,
    )
    labels = list(dataset.labels or ("x1", "x2"))
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(f"{dataset.name}: KRISP-U field-sampling trajectory")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"{dataset.name}_krispu_trajectory.png", dpi=200)
    plt.close(fig)


def _save_result_json(result: BenchmarkResult, output_dir: Path) -> None:
    payload = {
        "dataset": result.dataset_name,
        "objective": result.objective,
        "budget": result.budget,
        "n_initial": result.n_initial,
        "methods": {
            method: [
                {
                    "seed": trace.seed,
                    "best_y": trace.best_y,
                    "field_rmse": trace.field_rmse,
                    "field_mae": trace.field_mae,
                    "field_nrmse": trace.field_nrmse,
                    "field_nmae": trace.field_nmae,
                    "field_mape": trace.field_mape,
                    "field_r2": trace.field_r2,
                    "n_to_r2_threshold": trace.first_history_threshold_crossing(
                        "field_r2", R2_THRESHOLD
                    ),
                    "field_p95_abs_error": trace.field_p95_abs_error,
                    "field_max_abs_error": trace.field_max_abs_error,
                    "field_coverage_95": trace.field_coverage_95,
                    "mean_uncertainty": trace.mean_uncertainty,
                    "uncertainty_reduction": trace.uncertainty_reduction,
                    "field_r2_auc": trace.field_r2_auc,
                    "field_nrmse_auc": trace.field_nrmse_auc,
                    "selected_kernel_family": trace.selected_kernel_family,
                    "selected_kernel_repr": trace.selected_kernel_repr,
                    "kernel_family_history": trace.kernel_family_history,
                    "best_x": trace.best_x.tolist(),
                    "best_y_history": trace.best_y_history.tolist(),
                    "n_observed_history": _optional_array_to_list(
                        trace.n_observed_history
                    ),
                    "field_nrmse_history": _optional_array_to_list(
                        trace.field_nrmse_history
                    ),
                    "field_r2_history": _optional_array_to_list(trace.field_r2_history),
                    "field_p95_abs_error_history": _optional_array_to_list(
                        trace.field_p95_abs_error_history
                    ),
                    "field_max_abs_error_history": _optional_array_to_list(
                        trace.field_max_abs_error_history
                    ),
                    "field_coverage_95_history": _optional_array_to_list(
                        trace.field_coverage_95_history
                    ),
                    "mean_uncertainty_history": _optional_array_to_list(
                        trace.mean_uncertainty_history
                    ),
                }
                for trace in traces
            ]
            for method, traces in result.methods.items()
        },
    }
    path = output_dir / f"{result.dataset_name}_raw_traces.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _optional_array_to_list(values: np.ndarray | None) -> list[float] | None:
    if values is None:
        return None
    return values.tolist()


def _save_normalized_field_error_summary(
    summaries: list[dict[str, float | str]], output_dir: Path
) -> None:
    rows = [
        row
        for row in summaries
        if row["method"] in METHODS and np.isfinite(float(row["field_nrmse_mean"]))
    ]
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    methods = list(METHODS)
    x_axis = np.arange(len(datasets))
    width = _grouped_bar_width(methods)

    fig, ax = plt.subplots(figsize=(10, 5))
    for index, method in enumerate(methods):
        values = []
        for dataset in datasets:
            match = next(
                row
                for row in rows
                if row["dataset"] == dataset and row["method"] == method
            )
            values.append(float(match["field_nrmse_mean"]))
        ax.bar(
            x_axis + _grouped_bar_offset(index, methods, width),
            values,
            width=width,
            label=method,
        )
    ax.set_xticks(x_axis)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("mean final field NRMSE")
    ax.set_title("KRISP-U vs baseline reconstruction error, normalized per domain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "field_nrmse_summary.png", dpi=200)
    fig.savefig(output_dir / "field_rmse_summary.png", dpi=200)
    plt.close(fig)


def _save_r2_field_score_summary(
    summaries: list[dict[str, float | str]], output_dir: Path
) -> None:
    rows = [
        row
        for row in summaries
        if row["method"] in METHODS and np.isfinite(float(row["field_r2_mean"]))
    ]
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    methods = list(METHODS)
    x_axis = np.arange(len(datasets))
    width = _grouped_bar_width(methods)

    fig, ax = plt.subplots(figsize=(10, 5))
    for index, method in enumerate(methods):
        values = []
        for dataset in datasets:
            match = next(
                row
                for row in rows
                if row["dataset"] == dataset and row["method"] == method
            )
            values.append(float(match["field_r2_mean"]))
        ax.bar(
            x_axis + _grouped_bar_offset(index, methods, width),
            values,
            width=width,
            label=method,
        )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("mean final field R2")
    ax.set_title("KRISP-U vs baseline field variance explained")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "field_r2_summary.png", dpi=200)
    plt.close(fig)


def _save_domain_panel_plot(output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
    for ax, dataset_name in zip(axes.ravel(), DATASETS, strict=True):
        dataset = get_dataset(dataset_name)
        space = dataset.space()
        labels = list(dataset.labels or ("x1", "x2"))

        if isinstance(space, DiscreteCandidateSpace):
            candidates = space.candidates
            values = dataset.evaluate(candidates)
            image = ax.scatter(
                candidates[:, 0],
                candidates[:, 1],
                c=values,
                cmap="viridis",
                s=34,
                alpha=0.85,
            )
        else:
            x_axis = np.linspace(dataset.bounds[0, 0], dataset.bounds[0, 1], 160)
            y_axis = np.linspace(dataset.bounds[1, 0], dataset.bounds[1, 1], 160)
            grid_x, grid_y = np.meshgrid(x_axis, y_axis)
            points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
            values = dataset.evaluate(points).reshape(grid_x.shape)
            image = ax.contourf(grid_x, grid_y, values, levels=36, cmap="viridis")

        fig.colorbar(image, ax=ax, label="response", shrink=0.82)
        ax.set_title(dataset.name)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    fig.suptitle("Benchmark response fields")
    fig.savefig(output_dir / "benchmark_domains_2x3.png", dpi=220)
    plt.close(fig)


def _grouped_bar_width(methods: tuple[str, ...] | list[str]) -> float:
    return min(0.16, 0.82 / max(len(methods), 1))


def _grouped_bar_offset(
    index: int, methods: tuple[str, ...] | list[str], width: float
) -> float:
    return (index - (len(methods) - 1) / 2.0) * width


def _learning_summary_rows(
    result: BenchmarkResult,
    budgets: tuple[int, ...],
    n_bootstrap: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    metrics = (
        "field_nrmse",
        "field_r2",
        "field_p95_abs_error",
        "field_max_abs_error",
        "field_coverage_95",
        "mean_uncertainty",
    )
    for method, traces in result.methods.items():
        for budget in budgets:
            row: dict[str, float | str] = {
                "dataset": result.dataset_name,
                "budget": float(budget),
                "method": method,
            }
            for metric in metrics:
                values = np.asarray(
                    [_trace_history_value(trace, metric, budget) for trace in traces],
                    dtype=float,
                )
                low, high = _bootstrap_mean_ci_local(values, n_bootstrap=n_bootstrap)
                row[f"{metric}_mean"] = _nanmean_or_nan_local(values)
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            row["n_trials"] = float(len(traces))
            rows.append(row)
    return rows


def _r2_threshold_summary_rows(
    result: BenchmarkResult,
    threshold: float,
    n_bootstrap: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for method, traces in result.methods.items():
        threshold_counts = np.asarray(
            [
                _none_to_nan(
                    trace.first_history_threshold_crossing(
                        "field_r2",
                        threshold,
                    )
                )
                for trace in traces
            ],
            dtype=float,
        )
        final_r2_values = np.asarray(
            [trace.metric("field_r2") for trace in traces],
            dtype=float,
        )
        reached = np.isfinite(threshold_counts)
        reached_values = threshold_counts[reached]
        censored_values = np.where(reached, threshold_counts, float(result.budget))
        low, high = _bootstrap_mean_ci_local(
            reached_values,
            n_bootstrap=n_bootstrap,
        )
        censored_low, censored_high = _bootstrap_mean_ci_local(
            censored_values,
            n_bootstrap=n_bootstrap,
        )
        final_low, final_high = _bootstrap_mean_ci_local(
            final_r2_values,
            n_bootstrap=n_bootstrap,
        )
        rows.append(
            {
                "dataset": result.dataset_name,
                "method": method,
                "r2_threshold": float(threshold),
                "max_budget": float(result.budget),
                "reached_count": float(np.sum(reached)),
                "n_trials": float(len(traces)),
                "success_rate": float(np.mean(reached)),
                "n_to_threshold_mean": (
                    float(np.mean(reached_values))
                    if reached_values.size
                    else float("nan")
                ),
                "n_to_threshold_ci_low": low,
                "n_to_threshold_ci_high": high,
                "n_to_threshold_censored_mean": float(np.mean(censored_values)),
                "n_to_threshold_censored_ci_low": censored_low,
                "n_to_threshold_censored_ci_high": censored_high,
                "final_field_r2_mean": float(np.mean(final_r2_values)),
                "final_field_r2_ci_low": final_low,
                "final_field_r2_ci_high": final_high,
            }
        )
    return rows


def _none_to_nan(value: float | None) -> float:
    if value is None:
        return float("nan")
    return value


def _trace_history_value(trace: MethodTrace, metric: str, budget: int) -> float:
    n_observed = trace.history_metric("n_observed")
    values = trace.history_metric(metric)
    matches = np.flatnonzero(np.isclose(n_observed, float(budget)))
    if matches.size == 0:
        return float("nan")
    return float(values[int(matches[0])])


def _bootstrap_mean_ci_local(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 300,
    random_state: int = 0,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or n_bootstrap <= 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(random_state)
    sample_means = np.asarray(
        [
            np.mean(rng.choice(values, size=values.size, replace=True))
            for _ in range(n_bootstrap)
        ],
        dtype=float,
    )
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(sample_means, alpha)),
        float(np.quantile(sample_means, 1.0 - alpha)),
    )


def _nanmean_or_nan_local(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _finite_sweep_arrays(
    method_rows: list[dict[str, float | str]], metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    budgets = np.asarray([float(row["budget"]) for row in method_rows], dtype=float)
    means = np.asarray([float(row[f"{metric}_mean"]) for row in method_rows])
    lows = np.asarray([float(row[f"{metric}_ci_low"]) for row in method_rows])
    highs = np.asarray([float(row[f"{metric}_ci_high"]) for row in method_rows])
    finite = (
        np.isfinite(budgets)
        & np.isfinite(means)
        & np.isfinite(lows)
        & np.isfinite(highs)
    )
    return budgets[finite], means[finite], lows[finite], highs[finite]


def _set_domain_budget_xlim(
    ax: plt.Axes, dataset_rows: list[dict[str, float | str]], metric: str
) -> None:
    finite_budgets = [
        float(row["budget"])
        for row in dataset_rows
        if np.isfinite(float(row["budget"]))
        and np.isfinite(float(row[f"{metric}_mean"]))
    ]
    if not finite_budgets:
        return
    lower = min(finite_budgets)
    upper = max(finite_budgets)
    pad = max(0.5, 0.04 * max(upper - lower, 1.0))
    ax.set_xlim(max(0.0, lower - pad), upper + pad)


def _save_n_sweep_plot(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    if not rows:
        return

    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.5, 7.5),
        constrained_layout=True,
    )
    for ax, dataset in zip(axes.ravel(), datasets, strict=True):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for method in METHODS:
            method_rows = sorted(
                (row for row in dataset_rows if row["method"] == method),
                key=lambda row: float(row["budget"]),
            )
            budgets, means, lows, highs = _finite_sweep_arrays(
                method_rows, "field_nrmse"
            )
            if budgets.size == 0:
                continue
            ax.plot(budgets, means, marker="o", label=method)
            ax.fill_between(budgets, lows, highs, alpha=0.12)

        ax.set_title(dataset)
        ax.set_xlabel("total measured points (n)")
        ax.set_ylabel("field NRMSE")
        ax.set_ylim(bottom=0.0)
        _set_domain_budget_xlim(ax, dataset_rows, "field_nrmse")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Normalized reconstruction error vs measured points")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(METHODS),
    )
    fig.savefig(output_dir / "field_nrmse_vs_n.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_r2_sweep_plot(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    if not rows:
        return

    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.5, 7.5),
        constrained_layout=True,
    )
    for ax, dataset in zip(axes.ravel(), datasets, strict=True):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for method in METHODS:
            method_rows = sorted(
                (row for row in dataset_rows if row["method"] == method),
                key=lambda row: float(row["budget"]),
            )
            budgets, means, lows, highs = _finite_sweep_arrays(method_rows, "field_r2")
            if budgets.size == 0:
                continue
            ax.plot(budgets, means, marker="o", label=method)
            ax.fill_between(budgets, lows, highs, alpha=0.12)

        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.axhline(
            R2_THRESHOLD, color="black", linewidth=1.0, alpha=0.8, linestyle="--"
        )
        ax.set_title(dataset)
        ax.set_xlabel("total measured points (n)")
        ax.set_ylabel("field R2")
        _set_domain_budget_xlim(ax, dataset_rows, "field_r2")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Field variance explained vs measured points")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(METHODS),
    )
    fig.savefig(output_dir / "field_r2_vs_n.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_r2_threshold_plot(
    rows: list[dict[str, float | str]], output_dir: Path
) -> None:
    if not rows:
        return

    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    x_axis = np.arange(len(datasets))
    width = _grouped_bar_width(METHODS)

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for index, method in enumerate(METHODS):
        method_rows = [
            next(
                row
                for row in rows
                if row["dataset"] == dataset and row["method"] == method
            )
            for dataset in datasets
        ]
        means = np.asarray(
            [float(row["n_to_threshold_censored_mean"]) for row in method_rows]
        )
        lows = np.asarray(
            [
                max(
                    mean - float(row["n_to_threshold_censored_ci_low"]),
                    0.0,
                )
                for mean, row in zip(means, method_rows, strict=True)
            ]
        )
        highs = np.asarray(
            [
                max(
                    float(row["n_to_threshold_censored_ci_high"]) - mean,
                    0.0,
                )
                for mean, row in zip(means, method_rows, strict=True)
            ]
        )
        positions = x_axis + _grouped_bar_offset(index, METHODS, width)
        bars = ax.bar(
            positions,
            means,
            width=width,
            label=method,
            yerr=np.asarray([lows, highs]),
            capsize=3,
        )
        for bar, row in zip(bars, method_rows, strict=True):
            success_rate = float(row["success_rate"])
            if success_rate < 1.0:
                bar.set_hatch("//")
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 1.0,
                    f"{success_rate:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    max_budget = max(float(row["max_budget"]) for row in rows)
    ax.axhline(max_budget, color="black", linewidth=0.8, alpha=0.45)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel(f"measured points to field R2 >= {R2_THRESHOLD:.2f}")
    ax.set_title("Sample efficiency to reach target field reconstruction quality")
    ax.legend(title="Hatched = not all trials reached threshold")
    fig.savefig(output_dir / _r2_threshold_plot_filename(), dpi=220)
    plt.close(fig)


def _r2_threshold_plot_filename() -> str:
    return f"r2_{int(round(R2_THRESHOLD * 100)):02d}_points_to_threshold.png"


def _remove_stale_threshold_plots(output_dir: Path) -> None:
    current_name = _r2_threshold_plot_filename()
    for path in output_dir.glob("r2_*_points_to_threshold.png"):
        if path.name != current_name:
            path.unlink()


def _save_diagnostic_sweep_plot(
    rows: list[dict[str, float | str]],
    output_dir: Path,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    if not rows:
        return

    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.5, 7.5),
        constrained_layout=True,
    )
    for ax, dataset in zip(axes.ravel(), datasets, strict=True):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for method in METHODS:
            method_rows = sorted(
                (row for row in dataset_rows if row["method"] == method),
                key=lambda row: float(row["budget"]),
            )
            budgets, means, lows, highs = _finite_sweep_arrays(method_rows, metric)
            if budgets.size == 0:
                continue
            ax.plot(budgets, means, marker="o", label=method)
            ax.fill_between(budgets, lows, highs, alpha=0.12)

        if metric == "field_coverage_95":
            ax.axhline(0.95, color="black", linewidth=0.8, alpha=0.5)
            ax.set_ylim(0.0, 1.05)
        elif "error" in metric or metric == "mean_uncertainty":
            ax.set_ylim(bottom=0.0)
        ax.set_title(dataset)
        ax.set_xlabel("total measured points (n)")
        ax.set_ylabel(ylabel)
        _set_domain_budget_xlim(ax, dataset_rows, metric)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(title)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(METHODS),
    )
    fig.savefig(output_dir / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _lowest_error_trace(result: BenchmarkResult, method: str) -> MethodTrace:
    traces = result.methods[method]
    finite_traces = [trace for trace in traces if trace.field_rmse is not None]
    if not finite_traces:
        return traces[0]
    return min(finite_traces, key=lambda trace: float(trace.field_rmse))


def _primary_krispu_method(result: BenchmarkResult) -> str:
    if "krispu_adaptive" in result.methods:
        return "krispu_adaptive"
    if "krispu" in result.methods:
        return "krispu"
    return next(iter(result.methods))


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
