from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from benchmarks.kernel_study import run_kernel_selection_study
from krispu.kernels.builders import build_kernel_from_spec, build_named_kernel
from krispu.kernels.profiles import get_profile
from krispu.kernels.registry import KernelDefinition, registered_kernel_ids
from krispu.kernels.scoring import score_candidate_set, spatial_block_folds
from krispu.kernels.selection import KernelSelector


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.7]])
    return X, np.sin(X[:, 0]) + 0.3 * X[:, 1]


def test_manual_mode_constructs_requested_additive_kernel() -> None:
    kernel = build_kernel_from_spec(
        {
            "type": "additive",
            "components": [
                {"type": "rbf", "length_scale_initial": [0.4, 0.5]},
                {"type": "matern", "nu": 0.5, "length_scale_initial": [0.1, 0.1]},
            ],
        },
        2,
    )
    assert "RBF" in str(kernel)
    assert "nu=0.5" in str(kernel)


def test_manual_frozen_mode_disables_all_kernel_optimization() -> None:
    X, y = _data()
    result = KernelSelector(
        {
            "mode": "manual",
            "optimize_hyperparameters": False,
            "specification": {"type": "matern", "nu": 2.5},
        }
    ).select(X, y)
    assert all(parameter.fixed for parameter in result.fitted_kernel.hyperparameters)


def test_automatic_registry_contains_and_scores_every_initial_candidate() -> None:
    X, y = _data()
    scores = score_candidate_set(X, y, registered_kernel_ids(), optimizer_restarts=0)
    assert [score.candidate_kernel_id for score in scores] == list(registered_kernel_ids())
    assert all(np.isfinite(score.degeneracy_penalty) for score in scores)


def test_hybrid_registry_is_profile_limited() -> None:
    selector = KernelSelector({"mode": "hybrid", "profile": "rough_multiscale"})
    assert selector.allowed_kernel_ids == get_profile("rough_multiscale").allowed_kernel_ids
    assert "rbf_ard" not in selector.allowed_kernel_ids


def test_selection_api_has_no_acquisition_score_input() -> None:
    X, y = _data()
    result = KernelSelector({"mode": "hybrid", "profile": "smooth_global"}).select(X, y)
    assert result.selected_kernel_id in get_profile("smooth_global").allowed_kernel_ids


def test_candidate_failure_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    from krispu.kernels import registry

    definition = KernelDefinition(
        "bad_candidate",
        "bad",
        lambda dimension, optimize: (_ for _ in ()).throw(RuntimeError("bad")),
        1,
        "continuous",
        (),
        {},
        False,
    )
    monkeypatch.setitem(registry.KERNEL_REGISTRY, "bad_candidate", definition)
    score = score_candidate_set(*_data(), ["bad_candidate"])[0]
    assert not score.valid
    assert "fit failure" in score.penalty_reasons[0]


def test_multiscale_builder_keeps_long_and_short_semantics() -> None:
    kernel = build_named_kernel("matern_52_long_plus_matern_12_short", 2)
    parameters = kernel.get_params(deep=True)
    scales = [value for name, value in parameters.items() if name.endswith("length_scale")]
    assert np.allclose(scales[0], [0.6, 0.6])
    assert np.allclose(scales[1], [0.08, 0.08])


def test_spatial_block_folds_are_deterministic_and_separated() -> None:
    X, _ = _data()
    first = spatial_block_folds(X)
    second = spatial_block_folds(X)
    assert len(first) == len(second)
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))


def test_automatic_selection_returns_finite_predictive_diagnostics() -> None:
    result = KernelSelector(
        {"mode": "hybrid", "profile": "rough_single_scale", "minimum_points_before_selection": 6}
    ).select(*_data())
    assert np.isfinite(result.selection_score)
    assert all(np.isfinite(score.log_marginal_likelihood) for score in result.candidate_scores)


def test_hysteresis_rejects_an_insignificant_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    from krispu.kernels import selection as selection_module
    from krispu.kernels.scoring import CandidateScore

    X, y = _data()
    current = build_named_kernel("matern_32_ard", 2)
    challenger = build_named_kernel("rbf_ard", 2)

    calls = [0]

    def fake_scores(*args: object, **kwargs: object) -> list[CandidateScore]:
        calls[0] += 1
        current_score = 0.40 if calls[0] == 1 else 0.50
        challenger_score = 0.50 if calls[0] == 1 else 0.48
        return [
            CandidateScore(
                "matern_32_ard",
                "current",
                current_score,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                (),
                {},
                0,
                True,
                fitted_kernel=current,
            ),
            CandidateScore(
                "rbf_ard",
                "challenger",
                challenger_score,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                (),
                {},
                0,
                True,
                fitted_kernel=challenger,
            ),
        ]

    monkeypatch.setattr(selection_module, "score_candidate_set", fake_scores)
    selector = KernelSelector(
        {
            "mode": "automatic",
            "minimum_points_before_selection": 6,
            "minimum_score_improvement": 0.05,
        }
    )
    first = selector.select(X, y)
    X_next = np.vstack((X, [[0.8, 0.2], [0.1, 0.9], [0.7, 0.8]]))
    second = selector.select(X_next, np.sin(X_next[:, 0]) + 0.3 * X_next[:, 1])
    assert first.selected_kernel_id == second.selected_kernel_id == "matern_32_ard"
    assert not second.switch_accepted
    assert second.switch_rejection_reason == "challenger improvement below hysteresis threshold"


def test_manual_correct_and_mismatched_kernel_objects_are_distinct() -> None:
    correct = build_kernel_from_spec({"type": "matern", "nu": 2.5}, 2)
    mismatched = build_kernel_from_spec({"type": "rbf"}, 2)
    assert str(correct) != str(mismatched)


def test_kernel_study_writes_required_tables_and_figures(tmp_path: Path) -> None:
    config = {
        "study": "kernel_selection",
        "experiment_name": "kernel_selection_test",
        "fields": ["smooth"],
        "initial_sample_count": 5,
        "initial_boundary_margin": 0.05,
        "minimum_normalized_distance": 0.05,
        "final_budget": 6,
        "candidate_count": 8,
        "evaluation_grid_size": 5,
        "trials": 1,
        "base_seed": 5,
        "save_gifs": False,
        "kernel_selection": {"optimizer_restarts": 0, "reevaluate_every": 3},
    }
    config_path = tmp_path / "study.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = run_kernel_selection_study(config_path, tmp_path / "outputs")
    for name in (
        "kernel_candidate_scores.csv",
        "kernel_selection_history.csv",
        "kernel_hyperparameter_history.csv",
        "final_metrics.csv",
        "kernel_recovery_matrix.csv",
    ):
        assert (output / name).exists()
    assert list((output / "figures").glob("*.png"))
    with (output / "kernel_candidate_scores.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))
