"""Compare KRISP-U against random, grid, and LHS baselines."""

from __future__ import annotations

from krispu import run_benchmark


def main() -> None:
    result = run_benchmark(
        "branin",
        methods=("krispu", "random", "grid", "lhs"),
        budget=20,
        n_initial=5,
        n_trials=10,
        random_state=11,
        tolerance=1.0,
        n_candidates=1024,
    )
    print(result.summary())
    print(result.compare_to_baseline("krispu", "random"))
    print(result.compare_to_baseline("krispu", "grid"))


if __name__ == "__main__":
    main()
