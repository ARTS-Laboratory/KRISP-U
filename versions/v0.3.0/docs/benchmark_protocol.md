# Benchmark protocol (deferred)

The benchmark pass is intentionally deferred in the first implementation pass.
When added, paired trials must share the hidden field, domain, initial design,
responses, measurement budget, candidate restrictions, noise realization, and
recorded sub-seeds across every method. Every method must use the same
reconstruction model and field metrics such as RMSE, normalized RMSE, MAE,
`R^2`, p95 absolute error, and reconstruction-error AUC.

The primary comparison is field reconstruction, not objective regret. Random
fields and irregular domains must be included alongside smooth fields.
