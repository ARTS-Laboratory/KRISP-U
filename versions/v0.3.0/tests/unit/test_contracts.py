import numpy as np
import pytest

from krispu import (
    ContinuousDomain,
    GPRConfig,
    MixedDomain,
    ObservationSet,
    PolygonDomain,
    ResponseStandardizer,
)
from krispu.candidates import valid_candidate_mask


def test_coordinate_normalization_uses_declared_domain_bounds() -> None:
    domain = ContinuousDomain([[-10.0, 10.0], [100.0, 200.0]])
    point = np.array([[0.0, 150.0]])
    assert np.allclose(domain.normalize(point), [[0.5, 0.5]])
    assert np.allclose(domain.denormalize(domain.normalize(point)), point)


def test_observation_contract_requires_explicitly_valid_loo_mask() -> None:
    with pytest.raises(ValueError, match="one Boolean"):
        ObservationSet([[0, 0], [1, 1]], [0, 1], loo_eligible=[True])
    observations = ObservationSet(
        [[0, 0], [1, 1], [0, 1]], [0, 1, 2], loo_eligible=[True, False, True]
    )
    assert np.array_equal(observations.loo_eligible_indices, [0, 2])
    assert np.array_equal(observations.protected_indices, [1])


def test_response_standardization_is_invertible() -> None:
    standardizer = ResponseStandardizer.fit([2.0, 4.0, 8.0])
    transformed = standardizer.transform([2.0, 4.0, 8.0])
    assert np.allclose(standardizer.inverse_transform(transformed), [2.0, 4.0, 8.0])


def test_candidate_filter_removes_observed_and_excluded_points() -> None:
    domain = ContinuousDomain([[0, 1], [0, 1]])
    candidates = np.array([[0, 0], [0.5, 0.5], [1, 1], [1.2, 0.5]])
    mask = valid_candidate_mask(
        domain,
        candidates,
        [[0, 0]],
        excluded_regions=lambda points: np.all(points > 0.4, axis=1),
    )
    assert np.array_equal(mask, [False, False, False, False])


def test_discrete_options_are_not_continuously_validated() -> None:
    domain = MixedDomain([[0, 1]], [[10, 20]])
    assert np.array_equal(domain.contains([[0.5, 10], [0.5, 15]]), [True, False])


def test_polygon_membership_and_hole() -> None:
    domain = PolygonDomain(
        [[0, 0], [2, 0], [2, 2], [0, 2]],
        holes=[[[0.75, 0.75], [1.25, 0.75], [1.25, 1.25], [0.75, 1.25]]],
    )
    assert np.array_equal(domain.contains([[0.2, 0.2], [1, 1], [2.2, 1]]), [True, False, False])


def test_noise_configuration_rejects_ambiguous_deterministic_white_noise() -> None:
    with pytest.raises(ValueError, match="noisy mode"):
        GPRConfig(noise_mode="deterministic", fit_white_noise=True)
