# Candidate-level uncertainty

For each eligible observation `i`, the fixed-kernel LOO model predicts every
reference location `x`. The resulting field means are `f_-i(x)`. With `n`
eligible folds:

\[
\bar f_{LOO}(x)=\frac1n\sum_i f_{-i}(x),
\]

\[
U_{jack}(x)=\sqrt{\frac{n-1}{n}\sum_i
  [f_{-i}(x)-\bar f_{LOO}(x)]^2}.
\]

For the removed observation, the held-out residual and standardized residual
are `r_i=y_i-f_-i(X_i)` and `z_i=r_i/max(σ_-i(X_i), ε)`. The robust
calibration factor is

\[
c_{LOO}=\sqrt{\operatorname{median}(z_i^2)}.
\]

The calibrated posterior term and canonical KRISP-U uncertainty are

\[
U_{cal}(x)=c_{LOO}\sigma_{GP}(x),
\qquad
U_{KRISPU}(x)=\sqrt{U_{jack}(x)^2+U_{cal}(x)^2}.
\]

The API exposes the full-field means and standard deviations, held-out
predictions, residuals, standardized residuals, calibration factor, and all
three uncertainty components. Non-finite residuals are errors; they are not
silently replaced by zeros.

The reported GPR posterior standard deviation is latent-field predictive
uncertainty for the fitted covariance. In noisy mode, supplied observation
variances are included in the training covariance. A fitted white-noise
component is opt-in and deterministic mode has neither mechanism.
