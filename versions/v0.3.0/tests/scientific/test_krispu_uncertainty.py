from dataclasses import replace
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from benchmarks.evaluation import reconstruction_metrics
from benchmarks.visualization import panel_figure
from krispu import ContinuousDomain, GPRConfig, KrispURecommender, ObservationSet
from krispu.candidates import valid_candidate_mask


def _problem() -> tuple[ContinuousDomain, ObservationSet]:
    domain = ContinuousDomain([[-1.0, 1.0], [-1.0, 1.0]])
    X = np.array([[-0.6, -0.6], [0.6, -0.6], [-0.6, 0.6], [0.6, 0.6], [0.0, 0.0]])
    y = X[:, 0] ** 2 + 0.3 * X[:, 1]
    return domain, ObservationSet(X, y)


def _fixed_rbf(length_scale: float | tuple[float, float]) -> object:
    return ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
        length_scale=length_scale,
        length_scale_bounds="fixed",
    )


def _diagnostics(
    kernel: object,
    references: np.ndarray,
    *,
    alpha: float = 1.0e-10,
    noise_mode: str = "deterministic",
    observation_noise_variance: float | None = None,
):
    domain, observations = _problem()
    config = GPRConfig(
        kernel=kernel,
        optimize_hyperparameters=False,
        alpha=alpha,
        noise_mode=noise_mode,
        observation_noise_variance=observation_noise_variance,
    )
    return KrispURecommender(domain, gpr_config=config).evaluate_uncertainty(
        observations, references
    )


def test_raw_loo_field_sensitivity_can_be_nonzero_at_an_observed_point() -> None:
    _, observations = _problem()
    diagnostics = _diagnostics(_fixed_rbf(0.4), observations.X)
    assert np.any(diagnostics.loo_field_sensitivity > 1.0e-12)


def test_deterministic_observed_point_has_zero_support_deficit_and_krispu_uncertainty() -> None:
    _, observations = _problem()
    diagnostics = _diagnostics(_fixed_rbf(0.5), observations.X, alpha=1.0e-12)
    assert np.all(diagnostics.kernel_support_deficit < 1.0e-8)
    assert np.all(diagnostics.krispu_uncertainty < 1.0e-6)


def test_long_scale_kernel_suppresses_a_wider_neighborhood() -> None:
    references = np.array([[0.1, 0.0], [0.25, 0.0], [0.5, 0.0], [0.8, 0.0]])
    long = _diagnostics(_fixed_rbf(0.8), references)
    short = _diagnostics(_fixed_rbf(0.08), references)
    assert np.all(long.kernel_support_deficit[:2] < short.kernel_support_deficit[:2])
    assert np.sum(long.kernel_support_deficit < 0.5) >= np.sum(short.kernel_support_deficit < 0.5)


def test_ard_kernel_has_directional_support_suppression() -> None:
    references = np.array([[0.1, 0.0], [0.0, 0.1]])
    diagnostics = _diagnostics(_fixed_rbf((0.8, 0.08)), references)
    assert diagnostics.kernel_support_deficit[0] < diagnostics.kernel_support_deficit[1]


def test_noisy_observation_can_retain_local_support_deficit() -> None:
    _, observations = _problem()
    diagnostics = _diagnostics(
        _fixed_rbf(0.5),
        observations.X,
        noise_mode="noisy",
        observation_noise_variance=0.2,
    )
    assert np.any(diagnostics.kernel_support_deficit > 1.0e-4)


def test_krispu_formula_is_exact_and_independent_of_posterior_std() -> None:
    domain, _ = _problem()
    references = np.array([[-0.2, -0.2], [0.2, 0.2], [0.8, 0.0]])
    diagnostics = _diagnostics(_fixed_rbf(0.4), references)
    expected = diagnostics.loo_field_sensitivity * np.sqrt(diagnostics.kernel_support_deficit)
    assert np.array_equal(diagnostics.krispu_uncertainty, expected)
    altered = replace(diagnostics, posterior_std=np.linspace(100.0, 200.0, len(references)))
    assert np.array_equal(altered.krispu_uncertainty, diagnostics.krispu_uncertainty)
    assert domain.dimension == 2


def test_recommender_ranks_candidates_by_krispu_uncertainty() -> None:
    domain, observations = _problem()
    candidates = np.array([[-0.2, -0.2], [0.2, -0.2], [-0.2, 0.2], [0.2, 0.2]])
    recommender = KrispURecommender(domain, gpr_config=GPRConfig(kernel=_fixed_rbf(0.4)))
    diagnostics = recommender.evaluate_uncertainty(observations, candidates)
    result = recommender.recommend(observations, candidates=candidates)
    expected = int(np.argmax(diagnostics.krispu_uncertainty))
    assert np.allclose(result.as_array()[0], candidates[expected])
    assert result.recommendations[0].acquisition_score == diagnostics.krispu_uncertainty[expected]


def test_default_candidate_floor_is_small_and_only_rejects_near_duplicates() -> None:
    domain = ContinuousDomain([[0.0, 1.0], [0.0, 1.0]])
    observed = np.array([[0.5, 0.5]])
    candidates = np.array([[0.50001, 0.5], [0.5002, 0.5]])
    mask = valid_candidate_mask(domain, candidates, observed)
    assert np.array_equal(mask, [False, True])
    assert KrispURecommender(domain).min_normalized_distance == 1.0e-4


def test_panel_uses_final_krispu_uncertainty_for_the_uncertainty_panel() -> None:
    points = np.array([[x, y] for y in (0.0, 1.0) for x in (0.0, 1.0)])
    state = SimpleNamespace(
        evaluation_points=points,
        true_field=np.array([0.0, 1.0, 0.5, 1.5]),
        predicted_field=np.array([0.0, 0.9, 0.6, 1.4]),
        krispu_uncertainty=np.array([1.0, 2.0, 3.0, 4.0]),
        loo_field_sensitivity=np.array([100.0, 100.0, 100.0, 100.0]),
        posterior_std=np.array([50.0, 50.0, 50.0, 50.0]),
        metrics=reconstruction_metrics(
            np.array([0.0, 1.0, 0.5, 1.5]), np.array([0.0, 0.9, 0.6, 1.4])
        ),
        observed_X=np.array([[0.0, 0.0]]),
        initial_sample_count=1,
        recommended_point=np.array([1.0, 1.0]),
        annotate_point_order=True,
        current_length_scales=(0.5, 0.5),
        selection_mode="fixed_generic",
        selected_kernel_id="rbf_ard",
        method="support_adjusted_krispu",
        field="test",
        trial=0,
        sample_count=1,
    )
    figure = panel_figure(state)
    try:
        assert "KRISP-U uncertainty" in {axis.get_title() for axis in figure.axes}
    finally:
        plt.close(figure)
