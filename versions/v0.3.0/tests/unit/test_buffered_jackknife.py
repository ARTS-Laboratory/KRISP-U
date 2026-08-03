import numpy as np

from krispu.config import GPRConfig
from krispu.jackknife import BufferedJackknifePlan, build_buffered_jackknife_plan
from krispu.kernels.builders import build_named_kernel
from krispu.observations import ObservationSet
from krispu.surrogates.gpr import GPRSurrogate
from krispu.uncertainty.buffered_jackknife import compute_buffered_jackknife


def test_buffered_folds_remove_anchor_and_all_nearby_observations() -> None:
    points = np.array([[0.10, 0.10], [0.12, 0.10], [0.50, 0.50], [0.90, 0.90]])
    plan = build_buffered_jackknife_plan(points, multiplier=1.0, minimum_radius=0.025, maximum_radius=0.20, minimum_training_points=2)
    first = plan.removed_indices_by_fold[np.flatnonzero(plan.anchor_indices == 0)[0]]
    assert np.array_equal(first, [0, 1])


def test_noneligible_observations_are_removed_inside_an_eligible_buffer() -> None:
    points = np.array([[0.10, 0.10], [0.11, 0.10], [0.50, 0.50], [0.90, 0.90]])
    eligible = np.array([True, False, True, True])
    plan = build_buffered_jackknife_plan(points, eligible, multiplier=1.0, minimum_radius=0.025, maximum_radius=0.20, minimum_training_points=2)
    assert np.array_equal(plan.removed_indices_by_fold[0], [0, 1])


def test_fold_construction_is_deterministic_and_records_radius_reduction() -> None:
    points = np.array([[0.00], [0.01], [0.02], [0.03], [0.04]])
    kwargs = {
        "multiplier": 1.0,
        "minimum_radius": 0.025,
        "maximum_radius": 0.20,
        "minimum_training_points": 3,
    }
    first = build_buffered_jackknife_plan(points, **kwargs)
    second = build_buffered_jackknife_plan(points, **kwargs)
    assert np.array_equal(first.anchor_indices, second.anchor_indices)
    assert all(np.array_equal(left, right) for left, right in zip(first.removed_indices_by_fold, second.removed_indices_by_fold, strict=True))
    assert np.array_equal(first.effective_radius_by_fold, second.effective_radius_by_fold)
    assert first.effective_radius_by_fold[0] < first.global_buffer_radius
    assert first.training_count_by_fold[0] >= 3


def test_buffered_jackknife_uses_fixed_complete_fit_kernel() -> None:
    points = np.array([[0.0, 0.0], [0.2, 0.0], [0.8, 0.8], [1.0, 1.0]])
    values = points[:, 0] + points[:, 1]
    observations = ObservationSet(points, values)
    kernel = build_named_kernel("matern_32_ard", 2)
    surrogate = GPRSurrogate(GPRConfig(kernel=kernel, optimize_hyperparameters=False)).fit(points, values)
    plan = BufferedJackknifePlan.from_normalized_coordinates(points, minimum_radius=0.025, maximum_radius=0.20, minimum_training_points=2)
    result = compute_buffered_jackknife(surrogate, observations, points, plan)
    assert result.field_means.shape == (4, 4)
    assert np.all(np.isfinite(result.standardized_residuals))


def test_clustered_observations_do_not_collapse_to_pointwise_uncertainty() -> None:
    points = np.array([[0.10, 0.10], [0.105, 0.10], [0.11, 0.10], [0.90, 0.90], [0.10, 0.90]])
    plan = build_buffered_jackknife_plan(points, minimum_radius=0.025, maximum_radius=0.20, minimum_training_points=3)
    assert any(len(removed) > 1 for removed in plan.removed_indices_by_fold)
    assert any(radius > 0 for radius in plan.effective_radius_by_fold)
