# Migration from v0.2

Use `ContinuousDomain` or another v0.3.0 domain, create an `ObservationSet`,
and pass it to `KrispURecommender`. The v0.3.0 primary class is not an
optimizer and there is no `KRISPU = KrispUOptimizer` alias.

The default score is no longer posterior standard deviation, expected
improvement, UCB/LCB, or covariance-only information gain. It is the combined
candidate-level fixed-hyperparameter LOO uncertainty. The explicit
`posterior_std` option remains only as a named comparison baseline.

The old scalar-per-removed-point interpolation method is not part of the new
scientific core. Existing v0.1/v0.2 data and code are not modified by this
version.
