from __future__ import annotations

import numpy as np

from krispu import get_dataset, run_benchmark


def test_run_benchmark_summarizes_methods() -> None:
    result = run_benchmark(
        "quadratic_bowl_2d",
        methods=("krispu", "random", "grid"),
        budget=8,
        n_initial=5,
        n_trials=2,
        random_state=8,
        n_candidates=64,
        tolerance=0.25,
        optimize_continuous_acquisition=False,
    )
    summary = result.summary(n_bootstrap=20)
    comparison = result.compare_to_baseline("krispu", "random", n_bootstrap=20)

    assert set(summary) == {"krispu", "random", "grid"}
    assert np.isfinite(summary["krispu"]["best_y_mean"])
    assert np.isfinite(summary["krispu"]["field_rmse_mean"])
    assert np.isfinite(summary["krispu"]["field_mae_mean"])
    assert np.isfinite(summary["krispu"]["field_nrmse_mean"])
    assert np.isfinite(summary["krispu"]["field_nmae_mean"])
    assert np.isfinite(summary["krispu"]["field_mape_mean"])
    assert np.isfinite(summary["krispu"]["field_r2_mean"])
    assert np.isfinite(summary["krispu"]["field_p95_abs_error_mean"])
    assert np.isfinite(summary["krispu"]["field_max_abs_error_mean"])
    assert np.isfinite(summary["krispu"]["field_coverage_95_mean"])
    assert np.isfinite(summary["krispu"]["mean_uncertainty_mean"])
    assert np.isfinite(summary["krispu"]["uncertainty_reduction_mean"])
    assert np.isfinite(summary["krispu"]["field_r2_auc_mean"])
    assert np.isfinite(summary["krispu"]["field_nrmse_auc_mean"])
    assert comparison["n_pairs"] == 2

    trace = result.methods["krispu"][0]
    np.testing.assert_allclose(trace.history_metric("n_observed"), [5.0, 6.0, 7.0, 8.0])
    assert np.all(np.isfinite(trace.history_metric("field_nrmse")))
    assert np.all(np.isfinite(trace.history_metric("field_r2")))
    assert (
        trace.first_history_threshold_crossing(
            "field_r2",
            threshold=float(np.min(trace.history_metric("field_r2"))) - 1.0,
        )
        == 5.0
    )
    assert (
        trace.first_history_threshold_crossing(
            "field_r2",
            threshold=2.0,
        )
        is None
    )


def test_run_benchmark_hull_initial_design_starts_from_corners_plus_one() -> None:
    dataset = get_dataset("quadratic_bowl_2d")
    result = run_benchmark(
        dataset,
        methods=("krispu",),
        budget=7,
        n_initial=None,
        n_trials=1,
        random_state=12,
        n_candidates=64,
        optimize_continuous_acquisition=False,
        initial_design="hull",
    )
    trace = result.methods["krispu"][0]
    expected_corners = {
        (dataset.bounds[0, 0], dataset.bounds[1, 0]),
        (dataset.bounds[0, 0], dataset.bounds[1, 1]),
        (dataset.bounds[0, 1], dataset.bounds[1, 0]),
        (dataset.bounds[0, 1], dataset.bounds[1, 1]),
    }

    assert result.n_initial == 5
    assert expected_corners == {tuple(row) for row in trace.X[:4]}
    assert dataset.bounds[0, 0] < trace.X[4, 0] < dataset.bounds[0, 1]
    assert dataset.bounds[1, 0] < trace.X[4, 1] < dataset.bounds[1, 1]
    np.testing.assert_allclose(trace.history_metric("n_observed"), [5.0, 6.0, 7.0])


def test_run_benchmark_can_score_selected_learning_curve_points() -> None:
    result = run_benchmark(
        "quadratic_bowl_2d",
        methods=("krispu",),
        budget=12,
        n_initial=5,
        n_trials=1,
        random_state=18,
        n_candidates=64,
        optimize_continuous_acquisition=False,
        learning_curve_n_values=(5, 8, 12),
    )
    trace = result.methods["krispu"][0]

    np.testing.assert_allclose(trace.history_metric("n_observed"), [5.0, 8.0, 12.0])
    assert np.isfinite(trace.metric("field_r2"))


def test_run_benchmark_early_stops_individual_trace() -> None:
    result = run_benchmark(
        "quadratic_bowl_2d",
        methods=("krispu",),
        budget=20,
        n_initial=5,
        n_trials=1,
        random_state=20,
        n_candidates=64,
        optimize_continuous_acquisition=False,
        early_stop_metric="field_r2",
        early_stop_threshold=-10.0,
    )
    trace = result.methods["krispu"][0]

    assert len(trace.X) == 5
    np.testing.assert_allclose(trace.history_metric("n_observed"), [5.0])
    assert trace.first_history_threshold_crossing("field_r2", -10.0) == 5.0


def test_run_benchmark_adaptive_kernel_method_records_selected_kernel() -> None:
    result = run_benchmark(
        "quadratic_bowl_2d",
        methods=("krispu_adaptive",),
        budget=8,
        n_initial=5,
        n_trials=1,
        random_state=28,
        n_candidates=32,
        optimize_continuous_acquisition=False,
        early_stop_metric="field_r2",
        early_stop_threshold=0.99,
        adaptive_kernel=True,
    )
    trace = result.methods["krispu_adaptive"][0]

    assert trace.selected_kernel_family is not None
    assert trace.selected_kernel_repr is not None
    assert trace.kernel_family_history
    assert len(trace.X) <= 8
