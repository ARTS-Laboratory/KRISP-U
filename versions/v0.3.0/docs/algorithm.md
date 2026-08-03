# KRISP-U v0.3.0 algorithm

KRISP-U uses one global spatial covariance family and one globally shared
anisotropic length-scale vector at each sampling step. Coordinates are
normalized to the domain before fitting, distance calculations, buffering, or
kernel evaluation.

The canonical acquisition field is

```text
buffered-jackknife field sensitivity × sqrt(kernel support deficit)
```

Posterior standard deviation is retained as an evaluation diagnostic and
comparator. It is not multiplied into, calibrated into, or otherwise combined
with the canonical acquisition field.

## Sequential step

After the minimum fit size is reached, the selected family is optimized at
every step. Optimization warm-starts from the previous fit and updates the
amplitude, every axis of the global ARD scale vector, and the optional
observation-noise nugget. The complete observation fit is the only fit that
optimizes hyperparameters.

A deterministic `BufferedJackknifePlan` is then constructed once from the
normalized observations. Each eligible anchor removes itself and every
observation inside the global adaptive buffer, including observations that are
not eligible anchors. If that removes too many points, the radius is reduced
deterministically for that fold until `minimum_training_points` is met. The
effective radius is recorded. The same plan is used for every candidate family
at that sample count.

Each fold reuses the complete-fit kernel parameters, refits only the GP linear
algebra and response standardization, predicts the complete reference field,
and predicts the held-out anchor. The ensemble of fold field predictions gives
the buffered-jackknife field sensitivity. Kernel support is computed from the
selected complete-fit latent covariance, excluding the separate noise term.

Family reselection is triggered only by the configured first eligible check,
fit or validation failure, conditioning failure, repeated bound contact,
material score degradation, or the maximum check interval. When triggered,
all permitted families are optimized on the complete data and scored with the
same buffered plan. Buffered predictive log score is primary; upper-tail
normalized absolute error is the tie-breaker. A candidate switch is accepted
only when its relative improvement reaches the configured threshold. A check
that retains the current family is still recorded as a reselection event.
