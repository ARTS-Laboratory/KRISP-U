"""Field-reconstruction benchmark runners for KRISP-U evidence generation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from krispu.datasets import ToyDataset, get_dataset
from krispu.designs import corner_plus_interior_design
from krispu.metrics import best_so_far, simple_regret
from krispu.models import GprConfig, KernelPriorConfig, KernelPriorResult
from krispu.optimizer import KrispUOptimizer
from krispu.space import (
    ContinuousSpace,
    DiscreteCandidateSpace,
    HybridCandidateSpace,
    as_2d_float_array,
    make_rng,
)


@dataclass
class MethodTrace:
    """One method's observations for one benchmark seed."""

    method: str
    seed: int
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    objective: str
    optimum_y: float | None = None
    field_rmse: float | None = None
    field_mae: float | None = None
    field_nrmse: float | None = None
    field_nmae: float | None = None
    field_mape: float | None = None
    field_r2: float | None = None
    field_p95_abs_error: float | None = None
    field_max_abs_error: float | None = None
    field_coverage_95: float | None = None
    mean_uncertainty: float | None = None
    uncertainty_reduction: float | None = None
    field_r2_auc: float | None = None
    field_nrmse_auc: float | None = None
    n_observed_history: NDArray[np.float64] | None = None
    field_nrmse_history: NDArray[np.float64] | None = None
    field_r2_history: NDArray[np.float64] | None = None
    field_p95_abs_error_history: NDArray[np.float64] | None = None
    field_max_abs_error_history: NDArray[np.float64] | None = None
    field_coverage_95_history: NDArray[np.float64] | None = None
    mean_uncertainty_history: NDArray[np.float64] | None = None
    selected_kernel_family: str | None = None
    selected_kernel_repr: str | None = None
    kernel_family_history: list[str] = field(default_factory=list)

    @property
    def best_y_history(self) -> NDArray[np.float64]:
        return best_so_far(self.y, self.objective)

    @property
    def best_y(self) -> float:
        return float(self.best_y_history[-1])

    @property
    def best_x(self) -> NDArray[np.float64]:
        if self.objective == "minimize":
            index = int(np.argmin(self.y))
        else:
            index = int(np.argmax(self.y))
        return self.X[index].copy()

    @property
    def simple_regret_history(self) -> NDArray[np.float64] | None:
        if self.optimum_y is None:
            return None
        return simple_regret(self.best_y_history, self.optimum_y, self.objective)

    def metric(self, name: str) -> float:
        """Return a scalar summary metric for this trace."""

        name = name.lower()
        if name in {"best_y", "final_best"}:
            return self.best_y
        if name in {"auc", "best_auc"}:
            return float(np.mean(self.best_y_history))
        if name in {"regret", "final_regret", "simple_regret"}:
            regret = self.simple_regret_history
            return float("nan") if regret is None else float(regret[-1])
        if name in {"field_rmse", "final_field_rmse"}:
            return float("nan") if self.field_rmse is None else self.field_rmse
        if name in {"field_mae", "final_field_mae"}:
            return float("nan") if self.field_mae is None else self.field_mae
        if name in {"field_nrmse", "final_field_nrmse", "normalized_field_rmse"}:
            return float("nan") if self.field_nrmse is None else self.field_nrmse
        if name in {"field_nmae", "final_field_nmae", "normalized_field_mae"}:
            return float("nan") if self.field_nmae is None else self.field_nmae
        if name in {"field_mape", "final_field_mape", "mape"}:
            return float("nan") if self.field_mape is None else self.field_mape
        if name in {"field_r2", "final_field_r2", "r2"}:
            return float("nan") if self.field_r2 is None else self.field_r2
        if name in {"field_p95_abs_error", "p95_abs_error", "final_p95_abs_error"}:
            if self.field_p95_abs_error is None:
                return float("nan")
            return self.field_p95_abs_error
        if name in {"field_max_abs_error", "max_abs_error", "worst_region_error"}:
            if self.field_max_abs_error is None:
                return float("nan")
            return self.field_max_abs_error
        if name in {"field_coverage_95", "coverage_95"}:
            if self.field_coverage_95 is None:
                return float("nan")
            return self.field_coverage_95
        if name in {"mean_uncertainty", "integrated_uncertainty"}:
            return (
                float("nan") if self.mean_uncertainty is None else self.mean_uncertainty
            )
        if name in {"uncertainty_reduction", "integrated_uncertainty_reduction"}:
            if self.uncertainty_reduction is None:
                return float("nan")
            return self.uncertainty_reduction
        if name in {"field_r2_auc", "r2_auc"}:
            return float("nan") if self.field_r2_auc is None else self.field_r2_auc
        if name in {"field_nrmse_auc", "nrmse_auc"}:
            return (
                float("nan") if self.field_nrmse_auc is None else self.field_nrmse_auc
            )
        raise ValueError(f"Unknown trace metric: {name}")

    def history_metric(self, name: str) -> NDArray[np.float64]:
        """Return one learning-curve metric over observed-point counts."""

        name = name.lower()
        mapping = {
            "n_observed": self.n_observed_history,
            "field_nrmse": self.field_nrmse_history,
            "field_r2": self.field_r2_history,
            "field_p95_abs_error": self.field_p95_abs_error_history,
            "p95_abs_error": self.field_p95_abs_error_history,
            "field_max_abs_error": self.field_max_abs_error_history,
            "max_abs_error": self.field_max_abs_error_history,
            "worst_region_error": self.field_max_abs_error_history,
            "field_coverage_95": self.field_coverage_95_history,
            "coverage_95": self.field_coverage_95_history,
            "mean_uncertainty": self.mean_uncertainty_history,
            "integrated_uncertainty": self.mean_uncertainty_history,
        }
        values = mapping.get(name)
        if values is None:
            raise ValueError(f"Trace does not contain history metric: {name}")
        return values.copy()

    def first_history_threshold_crossing(
        self,
        metric: str,
        threshold: float,
        *,
        greater_is_better: bool = True,
    ) -> float | None:
        """Return the first observed-point count crossing a history threshold."""

        n_observed = self.history_metric("n_observed")
        values = self.history_metric(metric)
        finite = np.isfinite(values)
        if greater_is_better:
            crosses = finite & (values >= threshold)
        else:
            crosses = finite & (values <= threshold)
        indices = np.flatnonzero(crosses)
        if indices.size == 0:
            return None
        return float(n_observed[int(indices[0])])


@dataclass
class BenchmarkResult:
    """Collection of repeated benchmark traces."""

    dataset_name: str
    objective: str
    budget: int
    n_initial: int
    methods: dict[str, list[MethodTrace]] = field(default_factory=dict)
    optimum_y: float | None = None
    tolerance: float | None = None

    def summary(
        self,
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
        random_state: int | None = 0,
    ) -> dict[str, dict[str, float]]:
        """Return field-reconstruction and secondary optimization summaries."""

        output: dict[str, dict[str, float]] = {}
        for method, traces in self.methods.items():
            best_values = np.asarray(
                [trace.metric("best_y") for trace in traces], dtype=float
            )
            auc_values = np.asarray(
                [trace.metric("auc") for trace in traces], dtype=float
            )
            regret_values = np.asarray(
                [trace.metric("final_regret") for trace in traces], dtype=float
            )
            field_rmse_values = np.asarray(
                [trace.metric("field_rmse") for trace in traces], dtype=float
            )
            field_mae_values = np.asarray(
                [trace.metric("field_mae") for trace in traces], dtype=float
            )
            field_nrmse_values = np.asarray(
                [trace.metric("field_nrmse") for trace in traces], dtype=float
            )
            field_nmae_values = np.asarray(
                [trace.metric("field_nmae") for trace in traces], dtype=float
            )
            field_mape_values = np.asarray(
                [trace.metric("field_mape") for trace in traces], dtype=float
            )
            field_r2_values = np.asarray(
                [trace.metric("field_r2") for trace in traces], dtype=float
            )
            field_p95_values = np.asarray(
                [trace.metric("field_p95_abs_error") for trace in traces],
                dtype=float,
            )
            field_max_values = np.asarray(
                [trace.metric("field_max_abs_error") for trace in traces],
                dtype=float,
            )
            coverage_values = np.asarray(
                [trace.metric("field_coverage_95") for trace in traces], dtype=float
            )
            mean_uncertainty_values = np.asarray(
                [trace.metric("mean_uncertainty") for trace in traces], dtype=float
            )
            uncertainty_reduction_values = np.asarray(
                [trace.metric("uncertainty_reduction") for trace in traces],
                dtype=float,
            )
            field_r2_auc_values = np.asarray(
                [trace.metric("field_r2_auc") for trace in traces], dtype=float
            )
            field_nrmse_auc_values = np.asarray(
                [trace.metric("field_nrmse_auc") for trace in traces], dtype=float
            )
            low, high = bootstrap_mean_ci(
                best_values, confidence, n_bootstrap, random_state
            )
            field_low, field_high = bootstrap_mean_ci(
                field_rmse_values, confidence, n_bootstrap, random_state
            )
            normalized_field_low, normalized_field_high = bootstrap_mean_ci(
                field_nrmse_values, confidence, n_bootstrap, random_state
            )
            field_mape_low, field_mape_high = bootstrap_mean_ci(
                field_mape_values, confidence, n_bootstrap, random_state
            )
            field_r2_low, field_r2_high = bootstrap_mean_ci(
                field_r2_values, confidence, n_bootstrap, random_state
            )
            field_p95_low, field_p95_high = bootstrap_mean_ci(
                field_p95_values, confidence, n_bootstrap, random_state
            )
            field_max_low, field_max_high = bootstrap_mean_ci(
                field_max_values, confidence, n_bootstrap, random_state
            )
            coverage_low, coverage_high = bootstrap_mean_ci(
                coverage_values, confidence, n_bootstrap, random_state
            )
            uncertainty_low, uncertainty_high = bootstrap_mean_ci(
                mean_uncertainty_values, confidence, n_bootstrap, random_state
            )
            uncertainty_reduction_low, uncertainty_reduction_high = bootstrap_mean_ci(
                uncertainty_reduction_values,
                confidence,
                n_bootstrap,
                random_state,
            )
            r2_auc_low, r2_auc_high = bootstrap_mean_ci(
                field_r2_auc_values, confidence, n_bootstrap, random_state
            )
            nrmse_auc_low, nrmse_auc_high = bootstrap_mean_ci(
                field_nrmse_auc_values, confidence, n_bootstrap, random_state
            )
            if self.tolerance is not None and np.all(np.isfinite(field_rmse_values)):
                success_rate = float(np.mean(field_rmse_values <= self.tolerance))
            else:
                success_rate = float("nan")
            output[method] = {
                "best_y_mean": float(np.mean(best_values)),
                "best_y_ci_low": low,
                "best_y_ci_high": high,
                "auc_mean": float(np.mean(auc_values)),
                "field_rmse_mean": _nanmean_or_nan(field_rmse_values),
                "field_rmse_ci_low": field_low,
                "field_rmse_ci_high": field_high,
                "field_mae_mean": _nanmean_or_nan(field_mae_values),
                "field_nrmse_mean": _nanmean_or_nan(field_nrmse_values),
                "field_nrmse_ci_low": normalized_field_low,
                "field_nrmse_ci_high": normalized_field_high,
                "field_nmae_mean": _nanmean_or_nan(field_nmae_values),
                "field_mape_mean": _nanmean_or_nan(field_mape_values),
                "field_mape_ci_low": field_mape_low,
                "field_mape_ci_high": field_mape_high,
                "field_r2_mean": _nanmean_or_nan(field_r2_values),
                "field_r2_ci_low": field_r2_low,
                "field_r2_ci_high": field_r2_high,
                "field_p95_abs_error_mean": _nanmean_or_nan(field_p95_values),
                "field_p95_abs_error_ci_low": field_p95_low,
                "field_p95_abs_error_ci_high": field_p95_high,
                "field_max_abs_error_mean": _nanmean_or_nan(field_max_values),
                "field_max_abs_error_ci_low": field_max_low,
                "field_max_abs_error_ci_high": field_max_high,
                "field_coverage_95_mean": _nanmean_or_nan(coverage_values),
                "field_coverage_95_ci_low": coverage_low,
                "field_coverage_95_ci_high": coverage_high,
                "mean_uncertainty_mean": _nanmean_or_nan(mean_uncertainty_values),
                "mean_uncertainty_ci_low": uncertainty_low,
                "mean_uncertainty_ci_high": uncertainty_high,
                "uncertainty_reduction_mean": _nanmean_or_nan(
                    uncertainty_reduction_values
                ),
                "uncertainty_reduction_ci_low": uncertainty_reduction_low,
                "uncertainty_reduction_ci_high": uncertainty_reduction_high,
                "field_r2_auc_mean": _nanmean_or_nan(field_r2_auc_values),
                "field_r2_auc_ci_low": r2_auc_low,
                "field_r2_auc_ci_high": r2_auc_high,
                "field_nrmse_auc_mean": _nanmean_or_nan(field_nrmse_auc_values),
                "field_nrmse_auc_ci_low": nrmse_auc_low,
                "field_nrmse_auc_ci_high": nrmse_auc_high,
                "final_regret_mean": _nanmean_or_nan(regret_values),
                "success_rate": success_rate,
                "n_trials": float(len(traces)),
            }
        return output

    def compare_to_baseline(
        self,
        method: str = "krispu",
        baseline: str = "random",
        metric: str = "field_nrmse",
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
        random_state: int | None = 0,
    ) -> dict[str, float]:
        """Return bootstrap CI for paired method-minus-baseline differences.

        For field-error metrics such as ``field_nrmse``, negative differences
        mean KRISP-U had lower reconstruction error than the baseline. For R2,
        positive differences mean KRISP-U explained more field variance.
        """

        method_traces = sorted(self.methods[method], key=lambda trace: trace.seed)
        baseline_traces = sorted(self.methods[baseline], key=lambda trace: trace.seed)
        method_by_seed = {trace.seed: trace for trace in method_traces}
        baseline_by_seed = {trace.seed: trace for trace in baseline_traces}
        paired_seeds = sorted(set(method_by_seed) & set(baseline_by_seed))
        if not paired_seeds:
            raise ValueError("No shared seeds found for paired comparison.")
        differences = np.asarray(
            [
                method_by_seed[seed].metric(metric)
                - baseline_by_seed[seed].metric(metric)
                for seed in paired_seeds
            ],
            dtype=float,
        )
        differences = differences[np.isfinite(differences)]
        low, high = bootstrap_mean_ci(
            differences, confidence, n_bootstrap, random_state
        )
        return {
            "mean_difference": float(np.mean(differences)),
            "ci_low": low,
            "ci_high": high,
            "n_pairs": float(len(differences)),
        }


def bootstrap_mean_ci(
    values: ArrayLike,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: int | None = 0,
) -> tuple[float, float]:
    """Return a nonparametric bootstrap confidence interval for a mean."""

    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or n_bootstrap <= 1:
        value = float(values[0])
        return value, value
    rng = make_rng(random_state)
    sample_means = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        sample_means[index] = np.mean(sample)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(sample_means, alpha)),
        float(np.quantile(sample_means, 1.0 - alpha)),
    )


def _nanmean_or_nan(values: NDArray[np.float64]) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def run_benchmark(
    dataset: str | ToyDataset,
    methods: Iterable[str] = ("krispu", "random", "grid", "lhs"),
    budget: int = 25,
    n_initial: int | None = None,
    n_trials: int = 20,
    random_state: int = 0,
    acquisition: str = "uncertainty",
    n_candidates: int = 2048,
    tolerance: float | None = None,
    optimize_continuous_acquisition: bool = False,
    initial_design: str = "lhs",
    score_learning_curve: bool = True,
    learning_curve_n_values: Iterable[int] | None = None,
    early_stop_metric: str | None = None,
    early_stop_threshold: float | None = None,
    early_stop_greater_is_better: bool = True,
    adaptive_kernel: bool = True,
    kernel_prior_config: KernelPriorConfig | None = None,
) -> BenchmarkResult:
    """Run KRISP-U and baseline field-sampling methods with matched budgets."""

    dataset = get_dataset(dataset) if isinstance(dataset, str) else dataset
    initial_design = initial_design.lower().replace("-", "_")
    if n_initial is None and initial_design in {"hull", "corners", "corner_hull"}:
        n_initial = _hull_initial_count(dataset)
    else:
        n_initial = (
            n_initial
            or dataset.recommended_initial_n
            or max(4, 2 * dataset.dimension + 1)
        )
    if budget <= n_initial:
        raise ValueError("budget must be larger than n_initial.")
    if n_initial < 2:
        raise ValueError("n_initial must be at least 2.")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    if (early_stop_metric is None) != (early_stop_threshold is None):
        raise ValueError(
            "early_stop_metric and early_stop_threshold must be supplied together."
        )

    normalized_methods = tuple(method.lower() for method in methods)
    result = BenchmarkResult(
        dataset_name=dataset.name,
        objective=dataset.objective,
        budget=budget,
        n_initial=n_initial,
        optimum_y=dataset.optimum_y,
        tolerance=tolerance,
    )
    for method in normalized_methods:
        result.methods[method] = []

    for trial in range(n_trials):
        seed = random_state + trial
        initial_X = _initial_design(dataset, n_initial, initial_design, seed)
        initial_y = dataset.evaluate(initial_X)
        for method in normalized_methods:
            trace = _run_method(
                dataset=dataset,
                method=method,
                initial_X=initial_X,
                initial_y=initial_y,
                budget=budget,
                seed=seed,
                acquisition=acquisition,
                n_candidates=n_candidates,
                optimize_continuous_acquisition=optimize_continuous_acquisition,
                score_learning_curve=score_learning_curve,
                learning_curve_n_values=learning_curve_n_values,
                early_stop_metric=early_stop_metric,
                early_stop_threshold=early_stop_threshold,
                early_stop_greater_is_better=early_stop_greater_is_better,
                adaptive_kernel=adaptive_kernel,
                kernel_prior_config=kernel_prior_config,
            )
            result.methods[method].append(trace)
    return result


def estimate_oracle_best(
    dataset: ToyDataset,
    points_per_dimension: int = 250,
    n_random: int = 100_000,
    random_state: int = 0,
) -> tuple[NDArray[np.float64], float]:
    """Estimate a secondary reference best value from a dense candidate set."""

    space = dataset.space()
    if isinstance(space, DiscreteCandidateSpace):
        candidates = space.candidates
    elif dataset.dimension <= 2:
        candidates = space.dense_grid(points_per_dimension=points_per_dimension)
    else:
        candidates = space.sample(n_random, method="lhs", random_state=random_state)
    values = dataset.evaluate(candidates)
    if dataset.objective == "minimize":
        index = int(np.argmin(values))
    else:
        index = int(np.argmax(values))
    return candidates[index].copy(), float(values[index])


def _hull_initial_count(dataset: ToyDataset) -> int:
    if dataset.dimension >= 7:
        raise ValueError(
            "Hull initial design would create too many corner points for this "
            "dimension. Provide n_initial and a different initial_design."
        )
    return 2**dataset.dimension + 1


def _initial_design(
    dataset: ToyDataset,
    n_initial: int,
    initial_design: str,
    seed: int,
) -> NDArray[np.float64]:
    if initial_design in {"hull", "corners", "corner_hull"}:
        space = dataset.space()
        if not isinstance(space, ContinuousSpace):
            raise ValueError("Hull initial design requires a continuous space.")
        design = corner_plus_interior_design(dataset.bounds, random_state=seed)
        if len(design) != n_initial:
            raise ValueError(
                f"Hull initial design creates {len(design)} points for this "
                f"domain, but n_initial={n_initial}."
            )
        return design
    return dataset.initial_design(n_initial, method=initial_design, random_state=seed)


def _run_method(
    dataset: ToyDataset,
    method: str,
    initial_X: NDArray[np.float64],
    initial_y: NDArray[np.float64],
    budget: int,
    seed: int,
    acquisition: str,
    n_candidates: int,
    optimize_continuous_acquisition: bool,
    score_learning_curve: bool,
    learning_curve_n_values: Iterable[int] | None,
    early_stop_metric: str | None,
    early_stop_threshold: float | None,
    early_stop_greater_is_better: bool,
    adaptive_kernel: bool,
    kernel_prior_config: KernelPriorConfig | None,
) -> MethodTrace:
    n_needed = budget - len(initial_X)
    method_adaptive_kernel = _method_uses_adaptive_kernel(method, adaptive_kernel)
    scoring_gpr_config = GprConfig(
        n_restarts_optimizer=0,
        random_state=seed,
        adaptive_kernel=method_adaptive_kernel,
        kernel_prior_config=kernel_prior_config,
    )
    if early_stop_metric is not None and early_stop_threshold is not None:
        return _run_method_until_threshold(
            dataset=dataset,
            method=method,
            initial_X=initial_X,
            initial_y=initial_y,
            budget=budget,
            seed=seed,
            acquisition=acquisition,
            n_candidates=n_candidates,
            optimize_continuous_acquisition=optimize_continuous_acquisition,
            early_stop_metric=early_stop_metric,
            early_stop_threshold=early_stop_threshold,
            early_stop_greater_is_better=early_stop_greater_is_better,
            adaptive_kernel=method_adaptive_kernel,
            kernel_prior_config=kernel_prior_config,
            scoring_gpr_config=scoring_gpr_config,
        )

    if method in {
        "krispu",
        "krispu_fixed",
        "fixed_krispu",
        "krispu_adaptive",
        "adaptive_krispu",
    }:
        optimizer = KrispUOptimizer(
            dataset.space(),
            objective=dataset.objective,
            acquisition=acquisition,
            n_candidates=n_candidates,
            random_state=seed,
            optimize_continuous_acquisition=optimize_continuous_acquisition,
            gpr_config=scoring_gpr_config,
            kernel_prior_config=kernel_prior_config,
        )
        result = optimizer.run(
            dataset.evaluate,
            initial_X=initial_X,
            initial_y=initial_y,
            n_iterations=n_needed,
        )
        return MethodTrace(
            method=method,
            seed=seed,
            X=result.X,
            y=result.y,
            objective=dataset.objective,
            optimum_y=dataset.optimum_y,
            **_field_learning_metrics(
                dataset,
                result.X,
                result.y,
                seed,
                start_index=len(initial_X),
                score_learning_curve=score_learning_curve,
                learning_curve_n_values=learning_curve_n_values,
                scoring_gpr_config=scoring_gpr_config,
            ),
            **_kernel_trace_metadata(optimizer.model_selection_history_),
        )

    space = dataset.space()
    additions = _baseline_sequence(method, space, initial_X, n_needed, seed)

    X = np.vstack((initial_X, additions))
    y = dataset.evaluate(X)
    return MethodTrace(
        method=method,
        seed=seed,
        X=X,
        y=y,
        objective=dataset.objective,
        optimum_y=dataset.optimum_y,
        **_field_learning_metrics(
            dataset,
            X,
            y,
            seed,
            start_index=len(initial_X),
            score_learning_curve=score_learning_curve,
            learning_curve_n_values=learning_curve_n_values,
            scoring_gpr_config=GprConfig(
                n_restarts_optimizer=0,
                random_state=seed,
                adaptive_kernel=False,
            ),
        ),
    )


def _run_method_until_threshold(
    dataset: ToyDataset,
    method: str,
    initial_X: NDArray[np.float64],
    initial_y: NDArray[np.float64],
    budget: int,
    seed: int,
    acquisition: str,
    n_candidates: int,
    optimize_continuous_acquisition: bool,
    early_stop_metric: str,
    early_stop_threshold: float,
    early_stop_greater_is_better: bool,
    adaptive_kernel: bool,
    kernel_prior_config: KernelPriorConfig | None,
    scoring_gpr_config: GprConfig,
) -> MethodTrace:
    points = _field_evaluation_points(dataset, seed)
    true_values = dataset.evaluate(points)
    X = initial_X.copy()
    y = initial_y.copy()
    n_values: list[float] = []
    prefix_metrics: list[dict[str, float]] = []

    optimizer: KrispUOptimizer | None = None
    additions: NDArray[np.float64] | None = None
    if method in {
        "krispu",
        "krispu_fixed",
        "fixed_krispu",
        "krispu_adaptive",
        "adaptive_krispu",
    }:
        optimizer = KrispUOptimizer(
            dataset.space(),
            objective=dataset.objective,
            acquisition=acquisition,
            n_candidates=n_candidates,
            random_state=seed,
            optimize_continuous_acquisition=optimize_continuous_acquisition,
            gpr_config=scoring_gpr_config,
            kernel_prior_config=kernel_prior_config,
        )
    else:
        n_needed = budget - len(initial_X)
        additions = _baseline_sequence(
            method, dataset.space(), initial_X, n_needed, seed
        )

    def record_current_prefix() -> bool:
        metrics = _score_field_prefix(
            dataset,
            X,
            y,
            seed,
            points,
            true_values,
            gpr_config=scoring_gpr_config if adaptive_kernel else None,
        )
        prefix_metrics.append(metrics)
        n_values.append(float(len(X)))
        value = _prefix_metric_value(metrics, early_stop_metric)
        if not np.isfinite(value):
            return False
        if early_stop_greater_is_better:
            return value >= early_stop_threshold
        return value <= early_stop_threshold

    reached = record_current_prefix()
    addition_index = 0
    while not reached and len(X) < budget:
        if method in {
            "krispu",
            "krispu_fixed",
            "fixed_krispu",
            "krispu_adaptive",
            "adaptive_krispu",
        }:
            if optimizer is None:
                raise RuntimeError("KRISP-U optimizer was not initialized.")
            optimizer.fit(X, y)
            acquisition_result = optimizer.ask(acquisition=acquisition)
            next_x = acquisition_result.x_next.reshape(1, -1)
        else:
            if additions is None:
                raise RuntimeError("Baseline additions were not initialized.")
            next_x = additions[addition_index].reshape(1, -1)
            addition_index += 1
        next_y = dataset.evaluate(next_x)
        X = np.vstack((X, next_x))
        y = np.append(y, next_y)
        reached = record_current_prefix()

    return MethodTrace(
        method=method,
        seed=seed,
        X=X,
        y=y,
        objective=dataset.objective,
        optimum_y=dataset.optimum_y,
        **_learning_metrics_from_prefix_metrics(
            prefix_metrics,
            np.asarray(n_values, dtype=float),
        ),
        **_kernel_trace_metadata(
            [] if optimizer is None else optimizer.model_selection_history_
        ),
    )


def _baseline_sequence(
    method: str,
    space: ContinuousSpace | DiscreteCandidateSpace | HybridCandidateSpace,
    initial_X: NDArray[np.float64],
    n_needed: int,
    seed: int,
) -> NDArray[np.float64]:
    if method == "random":
        return _random_sequence(space, initial_X, n_needed, seed)
    elif method == "grid":
        return _grid_sequence(space, initial_X, n_needed)
    elif method in {"lhs", "latin_hypercube", "sobol"}:
        return _space_filling_sequence(space, initial_X, n_needed, method, seed)
    raise ValueError(f"Unknown benchmark method: {method}")


def _method_uses_adaptive_kernel(method: str, default: bool) -> bool:
    if method in {"krispu_adaptive", "adaptive_krispu"}:
        return True
    if method in {"krispu_fixed", "fixed_krispu"}:
        return False
    return bool(default)


def _kernel_trace_metadata(history: list[KernelPriorResult]) -> dict[str, object]:
    kernel_history = [
        result.selected_family
        for result in history
        if result.selected_family is not None
    ]
    if not history:
        return {}
    final_result = history[-1]
    return {
        "selected_kernel_family": final_result.selected_family,
        "selected_kernel_repr": final_result.selected_kernel_repr,
        "kernel_family_history": kernel_history,
    }


def _field_learning_metrics(
    dataset: ToyDataset,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    seed: int,
    start_index: int,
    score_learning_curve: bool,
    learning_curve_n_values: Iterable[int] | None,
    scoring_gpr_config: GprConfig | None = None,
) -> dict[str, float | NDArray[np.float64]]:
    points = _field_evaluation_points(dataset, seed)
    true_values = dataset.evaluate(points)
    if score_learning_curve:
        return _score_learning_curve(
            dataset,
            X,
            y,
            seed,
            points,
            true_values,
            start_index,
            learning_curve_n_values,
            scoring_gpr_config,
        )
    return _score_field_prefix(
        dataset, X, y, seed, points, true_values, gpr_config=scoring_gpr_config
    )


def _score_field_prefix(
    dataset: ToyDataset,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    seed: int,
    points: NDArray[np.float64],
    true_values: NDArray[np.float64],
    gpr_config: GprConfig | None = None,
) -> dict[str, float]:
    gpr_config = gpr_config or GprConfig(
        n_restarts_optimizer=0,
        random_state=seed,
        adaptive_kernel=False,
    )
    model = KrispUOptimizer(
        dataset.space(),
        objective=dataset.objective,
        acquisition="uncertainty",
        random_state=seed,
        optimize_continuous_acquisition=False,
        gpr_config=gpr_config,
    )
    model.fit(X, y)
    predicted, std = model.predict(points)
    errors = predicted - true_values
    response_range = float(np.max(true_values) - np.min(true_values))
    if response_range <= 1e-12:
        response_range = 1.0
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    denominator_floor = max(0.01 * response_range, 1e-12)
    denominator = np.maximum(np.abs(true_values), denominator_floor)
    mape = float(100.0 * np.mean(np.abs(errors) / denominator))
    sse = float(np.sum(errors**2))
    sst = float(np.sum((true_values - np.mean(true_values)) ** 2))
    r2 = float("nan") if sst <= 1e-12 else 1.0 - (sse / sst)
    coverage_95 = float(np.mean(np.abs(errors) <= 1.96 * np.maximum(std, 0.0)))
    return {
        "field_rmse": rmse,
        "field_mae": mae,
        "field_nrmse": rmse / response_range,
        "field_nmae": mae / response_range,
        "field_mape": mape,
        "field_r2": r2,
        "field_p95_abs_error": float(np.quantile(np.abs(errors), 0.95)),
        "field_max_abs_error": float(np.max(np.abs(errors))),
        "field_coverage_95": coverage_95,
        "mean_uncertainty": float(np.mean(std)),
    }


def _score_learning_curve(
    dataset: ToyDataset,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    seed: int,
    points: NDArray[np.float64],
    true_values: NDArray[np.float64],
    start_index: int,
    learning_curve_n_values: Iterable[int] | None = None,
    scoring_gpr_config: GprConfig | None = None,
) -> dict[str, float | NDArray[np.float64]]:
    if start_index < 2:
        raise ValueError("start_index must be at least 2 for GPR scoring.")
    if start_index > len(X):
        raise ValueError("start_index cannot exceed the number of observations.")

    n_indices = _learning_curve_indices(start_index, len(X), learning_curve_n_values)
    n_values = n_indices.astype(float)
    prefix_metrics = [
        _score_field_prefix(
            dataset,
            X[:n],
            y[:n],
            seed,
            points,
            true_values,
            gpr_config=scoring_gpr_config,
        )
        for n in n_indices
    ]
    return _learning_metrics_from_prefix_metrics(prefix_metrics, n_values)


def _learning_metrics_from_prefix_metrics(
    prefix_metrics: list[dict[str, float]],
    n_values: NDArray[np.float64],
) -> dict[str, float | NDArray[np.float64]]:
    final_metrics = dict(prefix_metrics[-1])
    mean_uncertainty_history = np.asarray(
        [metrics["mean_uncertainty"] for metrics in prefix_metrics], dtype=float
    )
    initial_uncertainty = float(mean_uncertainty_history[0])
    final_uncertainty = float(mean_uncertainty_history[-1])
    if initial_uncertainty > 1e-12:
        uncertainty_reduction = (
            initial_uncertainty - final_uncertainty
        ) / initial_uncertainty
    else:
        uncertainty_reduction = float("nan")

    r2_history = np.asarray(
        [metrics["field_r2"] for metrics in prefix_metrics], dtype=float
    )
    nrmse_history = np.asarray(
        [metrics["field_nrmse"] for metrics in prefix_metrics], dtype=float
    )
    final_metrics.update(
        {
            "uncertainty_reduction": float(uncertainty_reduction),
            "field_r2_auc": _normalized_curve_auc(n_values, r2_history),
            "field_nrmse_auc": _normalized_curve_auc(n_values, nrmse_history),
            "n_observed_history": n_values,
            "field_nrmse_history": nrmse_history,
            "field_r2_history": r2_history,
            "field_p95_abs_error_history": np.asarray(
                [metrics["field_p95_abs_error"] for metrics in prefix_metrics],
                dtype=float,
            ),
            "field_max_abs_error_history": np.asarray(
                [metrics["field_max_abs_error"] for metrics in prefix_metrics],
                dtype=float,
            ),
            "field_coverage_95_history": np.asarray(
                [metrics["field_coverage_95"] for metrics in prefix_metrics],
                dtype=float,
            ),
            "mean_uncertainty_history": mean_uncertainty_history,
        }
    )
    return final_metrics


def _prefix_metric_value(metrics: dict[str, float], name: str) -> float:
    name = name.lower()
    aliases = {
        "r2": "field_r2",
        "final_field_r2": "field_r2",
        "normalized_field_rmse": "field_nrmse",
        "final_field_nrmse": "field_nrmse",
        "final_field_rmse": "field_rmse",
        "final_field_mae": "field_mae",
        "p95_abs_error": "field_p95_abs_error",
        "final_p95_abs_error": "field_p95_abs_error",
        "max_abs_error": "field_max_abs_error",
        "worst_region_error": "field_max_abs_error",
        "coverage_95": "field_coverage_95",
        "integrated_uncertainty": "mean_uncertainty",
    }
    key = aliases.get(name, name)
    if key not in metrics:
        raise ValueError(f"Unknown early-stop metric: {name}")
    return float(metrics[key])


def _learning_curve_indices(
    start_index: int,
    end_index: int,
    requested_values: Iterable[int] | None,
) -> NDArray[np.int_]:
    if requested_values is None:
        return np.arange(start_index, end_index + 1, dtype=int)

    values = np.asarray(list(requested_values), dtype=int)
    if values.size == 0:
        raise ValueError("learning_curve_n_values cannot be empty.")
    if np.any(values < start_index) or np.any(values > end_index):
        raise ValueError(
            "learning_curve_n_values must fall between n_initial and budget."
        )
    values = np.concatenate((values, np.asarray([start_index, end_index], dtype=int)))
    return np.unique(values)


def _normalized_curve_auc(
    x_values: NDArray[np.float64], y_values: NDArray[np.float64]
) -> float:
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if x_values.size == 0:
        return float("nan")
    if x_values.size == 1 or np.isclose(x_values[-1], x_values[0]):
        return float(y_values[-1])
    return float(np.trapezoid(y_values, x_values) / (x_values[-1] - x_values[0]))


def _field_evaluation_points(
    dataset: ToyDataset,
    seed: int,
    points_per_dimension: int = 60,
    n_random: int = 5000,
) -> NDArray[np.float64]:
    space = dataset.space()
    if isinstance(space, DiscreteCandidateSpace):
        return space.candidates.copy()
    if dataset.dimension <= 2:
        return space.dense_grid(points_per_dimension=points_per_dimension)
    return space.sample(n_random, method="lhs", random_state=seed)


def _random_sequence(
    space: ContinuousSpace | DiscreteCandidateSpace | HybridCandidateSpace,
    observed: NDArray[np.float64],
    n_needed: int,
    seed: int,
) -> NDArray[np.float64]:
    if isinstance(space, DiscreteCandidateSpace):
        candidates = _remove_rows(space.candidates, observed)
        if len(candidates) < n_needed:
            raise ValueError("Not enough unused discrete candidates for the budget.")
        rng = make_rng(seed)
        indices = rng.choice(len(candidates), size=n_needed, replace=False)
        return candidates[indices]

    rows = []
    rng = make_rng(seed)
    while sum(len(batch) for batch in rows) < n_needed:
        batch = space.sample(max(n_needed * 2, 16), method="random", random_state=rng)
        batch = _remove_rows(batch, observed)
        if rows:
            batch = _remove_rows(batch, np.vstack(rows))
        rows.append(batch)
    return np.vstack(rows)[:n_needed]


def _space_filling_sequence(
    space: ContinuousSpace | DiscreteCandidateSpace | HybridCandidateSpace,
    observed: NDArray[np.float64],
    n_needed: int,
    method: str,
    seed: int,
) -> NDArray[np.float64]:
    if isinstance(space, DiscreteCandidateSpace):
        return _random_sequence(space, observed, n_needed, seed)
    n_pool = max(n_needed * 4, n_needed + 8)
    pool = space.sample(n_pool, method=method, random_state=seed)
    pool = _remove_rows(pool, observed)
    if len(pool) < n_needed:
        extra = _random_sequence(
            space,
            np.vstack((observed, pool)),
            n_needed - len(pool),
            seed + 99,
        )
        pool = np.vstack((pool, extra))
    return pool[:n_needed]


def _grid_sequence(
    space: ContinuousSpace | DiscreteCandidateSpace | HybridCandidateSpace,
    observed: NDArray[np.float64],
    n_needed: int,
) -> NDArray[np.float64]:
    if isinstance(space, DiscreteCandidateSpace):
        candidates = _lexicographic_order(_remove_rows(space.candidates, observed))
        if len(candidates) < n_needed:
            raise ValueError("Not enough unused discrete candidates for the budget.")
        return _evenly_spaced_rows(candidates, n_needed)

    points_per_dimension = max(
        2,
        int(np.ceil((n_needed + len(observed)) ** (1 / space.dimension))),
    )
    for _ in range(8):
        try:
            candidates = space.dense_grid(points_per_dimension=points_per_dimension)
        except ValueError:
            candidates = space.sample(n_needed * 10, method="lhs", random_state=0)
        candidates = _remove_rows(candidates, observed)
        if len(candidates) >= n_needed:
            return _evenly_spaced_rows(_lexicographic_order(candidates), n_needed)
        points_per_dimension += 1
    raise ValueError(
        "Unable to create enough grid candidates for the benchmark budget."
    )


def _remove_rows(
    candidates: ArrayLike, observed: ArrayLike, atol: float = 1e-10
) -> NDArray[np.float64]:
    candidate_array = as_2d_float_array(candidates, "candidates")
    observed_array = as_2d_float_array(observed, "observed")
    keep = np.ones(len(candidate_array), dtype=bool)
    for row in observed_array:
        keep &= ~np.all(np.isclose(candidate_array, row, atol=atol), axis=1)
    return candidate_array[keep]


def _lexicographic_order(values: NDArray[np.float64]) -> NDArray[np.float64]:
    keys = tuple(values[:, index] for index in reversed(range(values.shape[1])))
    return values[np.lexsort(keys)]


def _evenly_spaced_rows(values: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    if len(values) < n:
        raise ValueError("Not enough rows to sample.")
    if len(values) == n:
        return values.copy()
    indices = np.linspace(0, len(values) - 1, n, dtype=int)
    return values[indices].copy()
