from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from benchmarks.fields.smooth import smooth_field
from benchmarks.plotting import plot_field_audit
from benchmarks.runner import _initial_design
from krispu import ContinuousDomain, KrispURecommender, ObservationSet
from krispu.acquisition.loo_uncertainty import loo_uncertainty_scores
from krispu.sequential import _best_available, run_sequential_design


def _problem() -> tuple[ContinuousDomain, ObservationSet, np.ndarray]:
    domain = ContinuousDomain([[-1.0, 1.0], [-1.0, 1.0]])
    X = np.array([[-0.6, -0.6], [0.6, -0.6], [-0.6, 0.6], [0.6, 0.6], [0.0, 0.0]])
    y = X[:, 0] ** 2 + 0.3 * X[:, 1]
    candidates = np.array([[-0.2, -0.2], [0.2, -0.2], [-0.2, 0.2], [0.2, 0.2]])
    return domain, ObservationSet(X, y), candidates


def test_canonical_score_is_exactly_krispu_uncertainty() -> None:
    domain, observations, candidates = _problem()
    result = KrispURecommender(domain).recommend(observations, candidates=candidates)
    assert np.array_equal(
        loo_uncertainty_scores(result.diagnostics), result.diagnostics.krispu_uncertainty
    )
    assert (
        result.recommendations[0].acquisition_score == result.diagnostics.krispu_uncertainty.max()
    )


def test_posterior_std_and_legacy_combined_std_cannot_change_krispu_ranking() -> None:
    domain, observations, candidates = _problem()
    recommender = KrispURecommender(domain)
    original = recommender.evaluate_uncertainty(observations, candidates)
    altered = replace(
        original,
        posterior_std=original.posterior_std[::-1].copy(),
        combined_std=np.linspace(100.0, 200.0, len(candidates)),
    )
    recommender.evaluate_uncertainty = lambda *_args: altered  # type: ignore[method-assign]
    result = recommender.recommend(observations, candidates=candidates)
    expected = int(np.argmax(original.krispu_uncertainty))
    assert np.allclose(result.as_array()[0], candidates[expected])
    assert result.recommendations[0].acquisition_score == original.krispu_uncertainty[expected]


def test_best_available_uses_observations_and_minimum_distance() -> None:
    domain = ContinuousDomain([[0.0, 1.0], [0.0, 1.0]])
    pool = np.array([[0.51, 0.5], [0.9, 0.9], [0.2, 0.8]])
    available = np.ones(len(pool), dtype=bool)
    scores = np.array([100.0, 2.0, 1.0])
    index = _best_available(pool, available, scores, domain, np.array([[0.5, 0.5]]), 0.05)
    assert index == 1
    observed = np.vstack(([[0.5, 0.5]], pool[1]))
    index = _best_available(pool, available, scores, domain, observed, 0.05)
    assert index == 2


def test_interior_initial_design_has_no_exact_corners_and_is_paired() -> None:
    domain = ContinuousDomain([[-1.0, 1.0], [-1.0, 1.0]])
    first, first_mask = _initial_design(
        "interior_maximin", domain, 5, 0.05, 17, return_eligibility=True
    )
    second, second_mask = _initial_design(
        "interior_maximin", domain, 5, 0.05, 17, return_eligibility=True
    )
    corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    assert not np.any(np.all(np.isclose(first[:, None], corners[None, :]), axis=2))
    assert np.array_equal(first, second)
    assert np.array_equal(first_mask, second_mask)
    field = smooth_field()
    candidates = np.array([[-0.8, 0.0], [0.8, 0.0], [0.0, -0.8], [0.0, 0.8]])
    runs = [
        run_sequential_design(
            field.evaluate,
            field.domain,
            first,
            candidates,
            first,
            method,
            6,
            21,
            initial_loo_eligible=first_mask,
        )
        for method in ("krispu_loo", "posterior_std")
    ]
    assert np.array_equal(runs[0][0].observed_X, runs[1][0].observed_X)


def test_anchor_eligibility_is_preserved_and_new_points_are_eligible() -> None:
    field = smooth_field()
    initial = np.array([[0.0, -1.0], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    mask = np.array([False, True, True, True, True])
    candidates = np.array([[-0.8, 0.0], [0.8, 0.0], [0.0, 0.8]])
    evaluation = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5]])
    states = run_sequential_design(
        field.evaluate,
        field.domain,
        initial,
        candidates,
        evaluation,
        "krispu_loo",
        7,
        3,
        initial_loo_eligible=mask,
        minimum_normalized_distance=0.05,
    )
    assert all(np.array_equal(state.observed_loo_eligible[:5], mask) for state in states)
    assert all(
        np.all(state.observed_loo_eligible[5:]) for state in states if state.sample_count > 5
    )


def test_sequential_selections_are_unique_and_record_geometry() -> None:
    field = smooth_field()
    initial = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.0, 0.0]])
    candidates = np.array([[-0.95, 0.0], [0.0, 0.95], [0.85, 0.0], [0.0, -0.85]])
    states = run_sequential_design(
        field.evaluate,
        field.domain,
        initial,
        candidates,
        initial,
        "random",
        7,
        11,
        minimum_normalized_distance=0.05,
        boundary_margin=0.05,
    )
    selected = states[-1].observed_X[5:]
    assert len(np.unique(np.round(selected, 12), axis=0)) == len(selected)
    assert all(state.distance_to_nearest_observation is not None for state in states[:-1])
    assert all(state.distance_to_domain_boundary is not None for state in states[:-1])
    assert all(state.on_current_sample_hull is not None for state in states[:-1])


def test_benchmark_config_and_field_plot_use_no_combined_panel() -> None:
    for name in ("paired_smoke.yaml", "visual_audit.yaml"):
        config = yaml.safe_load(Path("benchmarks/configs", name).read_text(encoding="utf-8"))
        assert "krispu_combined" not in config["methods"]
        assert "krispu_jackknife" not in config["methods"]
        assert config["methods"] == [
            "raw_loo_sensitivity",
            "support_adjusted_krispu",
            "posterior_std",
            "random",
            "lhs",
            "maximin",
        ]
    assert "combined" not in inspect.getsource(plot_field_audit).lower()
