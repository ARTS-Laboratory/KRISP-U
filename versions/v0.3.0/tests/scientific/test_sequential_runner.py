import numpy as np

from benchmarks.fields.smooth import smooth_field
from krispu.sequential import run_sequential_design


def test_sequential_runner_records_shared_observations() -> None:
    field = smooth_field()
    initial = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    candidates = np.array([[-0.8, -0.8], [0.8, -0.8], [-0.8, 0.8], [0.8, 0.8], [0.1, 0.3]])
    evaluation = np.array([[x, y] for x in np.linspace(-1, 1, 5) for y in np.linspace(-1, 1, 5)])
    runs = {
        method: run_sequential_design(
            field.evaluate,
            field.domain,
            initial,
            candidates,
            evaluation,
            method,
            7,
            9,
            field_name="smooth",
        )
        for method in ("krispu_combined", "posterior_std", "random")
    }
    for states in runs.values():
        assert [state.sample_count for state in states] == list(range(5, 8))
        selected = np.vstack([state.observed_X for state in states])
        assert len(np.unique(np.round(selected, 12), axis=0)) == len(
            np.unique(np.round(states[-1].observed_X, 12), axis=0)
        )
        assert len(states[-1].predicted_field) == len(evaluation)
        for state in states:
            assert np.allclose(state.observed_y, field.evaluate(state.observed_X))
    assert runs["krispu_combined"][0].jackknife_std is not None
    assert runs["krispu_combined"][0].combined_std is not None
    assert runs["random"][0].jackknife_std is None
    assert runs["random"][0].combined_std is None


def test_methods_use_candidate_pool_without_repetition() -> None:
    field = smooth_field()
    initial = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    candidates = np.array(
        [[-0.9, -0.9], [0.9, -0.9], [-0.9, 0.9], [0.9, 0.9], [0.2, 0.1], [-0.2, 0.1]]
    )
    evaluation = candidates.copy()
    states = run_sequential_design(
        field.evaluate, field.domain, initial, candidates, evaluation, "lhs", 7, 4
    )
    selected = states[-1].observed_X[5:]
    assert len(np.unique(np.round(selected, 12), axis=0)) == len(selected)
    assert all(np.any(np.all(np.isclose(candidates, point), axis=1)) for point in selected)
