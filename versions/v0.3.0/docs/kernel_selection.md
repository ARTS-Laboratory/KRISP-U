# Kernel selection in v0.3.0

Kernel selection changes only the reconstruction model used to generate the
LOO fields. The canonical KRISP-U acquisition remains the candidate-level LOO
field uncertainty; no posterior standard deviation is added to that score.

## Modes and registry

`manual` parses the explicit YAML schema in `krispu.kernels.builders` and
preserves the requested family. Bounds are normalized-coordinate bounds.
`optimize_hyperparameters: false` sets all manually supplied kernel bounds to
`fixed` and disables the GPR optimizer. `automatic` uses the finite registry in
`krispu.kernels.registry`; it does not build arbitrary expression trees.
`hybrid` restricts that registry using a named profile in
`krispu.kernels.profiles`.

The initial registry is deliberately small: five single-scale, four
long-plus-short multiscale, two trend, and two periodic candidates. The
multiscale builders place the long component first with bounds `[0.15, 2.0]`
and the short component second with bounds `[0.01, 0.20]`. A fitted candidate
is penalized when the short scale is not smaller than the long scale.

## Predictive scores

For every candidate, a full observation fit is optimized once. Each LOO or
spatial-block fold then clones that fitted kernel with `optimizer=None`; fold
hyperparameters are never re-optimized. With held-out residual `e_i` and
predictive standard deviation `s_i`, the reported metrics are

```text
NRMSE = sqrt(mean(e_i^2)) / (max(y) - min(y))
NLPD  = mean(0.5*log(2*pi*s_i^2) + e_i^2/(2*s_i^2))
calibration error = abs(mean((e_i/s_i)^2) - 1)
```

`loo_predictive` uses the LOO metrics. `spatial_block_cv` uses deterministic
occupied quadrants for two-dimensional normalized coordinates and falls back
to LOO when fewer than four observations or fewer than two occupied blocks
are available. `spatial_cv_composite` is the default. It min-max normalizes
each spatial-CV metric across the valid candidate set and minimizes

```text
0.5 * normalized spatial NLPD
+ 0.4 * normalized spatial NRMSE
+ 0.1 * normalized calibration error
+ degeneracy penalty
```

Marginal likelihood is recorded for comparison but is never the sole default
selection criterion. Candidate failures and spatial-CV fallback are recorded
in `kernel_candidate_scores.csv`.

## Hysteresis and benchmark studies

The deterministic profile default is used below six observations. Candidates
are reevaluated every three new measurements by default, and a challenger
must improve the lower-is-better score by at least `0.05` to replace the
current family. Study A compares complete workflows. Study B replays the
selected family at each sample count across `krispu_loo`, posterior standard
deviation, random, LHS, and maximin to isolate acquisition behavior.
