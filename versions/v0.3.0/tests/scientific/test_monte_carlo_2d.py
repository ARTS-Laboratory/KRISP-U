import numpy as np

from evaluation.runners.monte_carlo_2d import (
    FIELD_NAMES,
    METHOD_NAMES,
    AdaptiveKernelController,
    _jackknife_uncertainty_acquisition,
    _make_trial_field,
    _raw_jackknife_uncertainty_acquisition,
)
from krispu.config import GPRConfig
from krispu.jackknife import jackknife_field_sensitivity
from krispu.jackknife.plan import build_buffered_jackknife_plan
from krispu.surrogates.gpr import GPRSurrogate


def test_monte_carlo_includes_support_agnostic_adaptive_kriging() -> None:
    assert "adaptive_kriging" in METHOD_NAMES


def test_matched_spherical_case_uses_the_kriging_family() -> None:
    assert "matched_spherical" in FIELD_NAMES
    field = _make_trial_field(
        "matched_spherical",
        {
            "rotation_degrees": 0.0,
            "axis_scale": (1.0, 1.0),
            "translation": (0.0, 0.0),
            "output_scale": 1.0,
        },
        noise_seed=19,
    )

    values = field.clean(np.array([[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]]))

    assert field.metadata["true_kernel"]["family"] == "spherical_ard"
    assert field.metadata["kriging_kernel_family"] == "spherical_ard"
    assert np.all(np.isfinite(values))


def test_matched_spherical_controller_keeps_the_kernel_family_fixed() -> None:
    points = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    values = np.array([-1.0, 0.5, 0.75, -0.25])
    controller = AdaptiveKernelController(
        17,
        GPRConfig(optimize_hyperparameters=False, random_state=17),
        initial_family="spherical_ard",
        fixed_family="spherical_ard",
    )

    outcome = controller.select(points, values, sample_count=len(points))

    assert outcome.family == "spherical_ard"
    assert outcome.switch_accepted is False


def test_adaptive_kriging_acquisition_is_jackknife_only() -> None:
    points = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    values = np.array([-1.0, 0.5, 0.75, -0.25])
    references = np.array(
        [[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5], [0.9, -0.2], [-0.2, 0.8]]
    )
    surrogate = GPRSurrogate(
        GPRConfig(optimize_hyperparameters=False, random_state=17)
    ).fit(points, values)

    plan = build_buffered_jackknife_plan(
        points,
        multiplier=1.0,
        minimum_radius=0.025,
        maximum_radius=0.20,
        minimum_training_points=3,
    )
    evaluation, candidates = _jackknife_uncertainty_acquisition(
        surrogate,
        points,
        values,
        references,
        plan,
        evaluation_count=2,
    )
    fold_means = []
    for removed in plan.removed_indices_by_fold:
        keep = np.ones(len(points), dtype=bool)
        keep[removed] = False
        fold = GPRSurrogate(surrogate.config).fit_fixed_kernel(
            points[keep], values[keep], frozen_kernel=surrogate.frozen_kernel
        )
        mean, _ = fold.predict(references)
        fold_means.append(mean)
    _, expected = jackknife_field_sensitivity(np.column_stack(fold_means))

    np.testing.assert_allclose(evaluation, expected[:2])
    np.testing.assert_allclose(candidates, expected[2:])


def test_raw_jackknife_acquisition_uses_leave_one_out_folds() -> None:
    points = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    values = np.array([-1.0, 0.5, 0.75, -0.25])
    references = np.array(
        [[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5], [0.9, -0.2], [-0.2, 0.8]]
    )
    surrogate = GPRSurrogate(
        GPRConfig(optimize_hyperparameters=False, random_state=17)
    ).fit(points, values)

    evaluation, candidates = _raw_jackknife_uncertainty_acquisition(
        surrogate,
        points,
        values,
        references,
        evaluation_count=2,
    )
    fold_means = []
    for removed_index in range(len(points)):
        keep = np.ones(len(points), dtype=bool)
        keep[removed_index] = False
        fold = GPRSurrogate(surrogate.config).fit_fixed_kernel(
            points[keep], values[keep], frozen_kernel=surrogate.frozen_kernel
        )
        mean, _ = fold.predict(references)
        fold_means.append(mean)
    _, expected = jackknife_field_sensitivity(np.column_stack(fold_means))

    np.testing.assert_allclose(evaluation, expected[:2])
    np.testing.assert_allclose(candidates, expected[2:])
