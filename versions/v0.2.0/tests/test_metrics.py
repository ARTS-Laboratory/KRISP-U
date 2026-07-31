from __future__ import annotations

import numpy as np

from krispu.metrics import best_so_far, jsd, kld, mse, simple_regret


def test_metrics_are_finite() -> None:
    y_true = np.asarray([1.0, 2.0, 3.0])
    y_pred = np.asarray([1.0, 2.5, 2.5])

    assert mse(y_true, y_pred) > 0
    assert np.isfinite(kld(y_true, y_pred))
    assert np.isfinite(jsd(y_true, y_pred))


def test_best_so_far_and_regret_for_minimization() -> None:
    values = np.asarray([5.0, 3.0, 4.0, 2.0])
    best = best_so_far(values, objective="minimize")

    np.testing.assert_allclose(best, [5.0, 3.0, 3.0, 2.0])
    np.testing.assert_allclose(
        simple_regret(best, 1.5, "minimize"), [3.5, 1.5, 1.5, 0.5]
    )
