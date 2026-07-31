"""Basic KRISP-U field sampling on a 2D toy problem."""

from __future__ import annotations

from krispu import KrispUOptimizer, get_dataset


def main() -> None:
    dataset = get_dataset("branin")
    initial_X = dataset.initial_design(n=6, random_state=7)
    initial_y = dataset.evaluate(initial_X)

    optimizer = KrispUOptimizer(
        dataset.space(),
        objective=dataset.objective,
        acquisition="uncertainty",
        random_state=7,
        n_candidates=1024,
    )
    result = optimizer.run(
        dataset.evaluate,
        initial_X=initial_X,
        initial_y=initial_y,
        n_iterations=12,
    )

    print(f"Measured points: {len(result.X)}")
    print(f"Last selected point: {result.acquisitions[-1].x_next}")


if __name__ == "__main__":
    main()
