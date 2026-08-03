import numpy as np
import pytest

from evaluation.methods import lhs_order, maximin_index, random_order
from evaluation.metrics import nrmse_auc, paired_difference, reconstruction_metrics
from krispu.domains import ContinuousDomain


def test_known_metrics() -> None:
    metrics = reconstruction_metrics([0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 1.0, 3.0])
    assert metrics.rmse == pytest.approx(0.70710678)
    assert metrics.nrmse == pytest.approx(0.23570226)
    assert metrics.mae == pytest.approx(0.5)
    assert metrics.nmae == pytest.approx(1.0 / 6.0)
    assert metrics.p95_absolute_error == pytest.approx(1.0)
    assert metrics.max_absolute_error == pytest.approx(1.0)


def test_constant_range_fails_clearly() -> None:
    with pytest.raises(ValueError, match="constant"):
        reconstruction_metrics([1.0, 1.0], [1.0, 2.0])


def test_selection_sequences_are_deterministic() -> None:
    pool = np.array([[0.0, 0.0], [0.2, 0.4], [0.5, 0.5], [0.9, 0.1]])
    domain = ContinuousDomain([[0.0, 1.0], [0.0, 1.0]])
    assert np.array_equal(random_order(4, 11), random_order(4, 11))
    assert np.array_equal(lhs_order(pool, domain, 3, 11), lhs_order(pool, domain, 3, 11))


def test_maximin_selects_farthest_candidate() -> None:
    pool = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    observed = np.array([[0.5, 0.5]])
    domain = ContinuousDomain([[0.0, 1.0], [0.0, 1.0]])
    index = maximin_index(pool, observed, domain, np.array([True, True, False]))
    assert index == 0


def test_nrmse_auc_uses_right_endpoint_increments() -> None:
    assert nrmse_auc([5, 6, 8], [2.0, 3.0, 4.0]) == pytest.approx(11.0)


def test_paired_difference_preserves_trial_alignment() -> None:
    assert np.allclose(paired_difference([0.2, 0.4], [0.3, 0.1]), [-0.1, 0.3])
