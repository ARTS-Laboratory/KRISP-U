# KRISP-U uncertainty

For each eligible observation, the fixed fitted-kernel LOO model predicts the
complete reference field. With `n` folds,

\[
\bar f_{LOO}(x)=\frac1n\sum_i\hat f_{-i}(x),
\qquad
S_{LOO}(x)=\sqrt{\frac{n-1}{n}\sum_i[\hat f_{-i}(x)-\bar f_{LOO}(x)]^2}.
\]

`S_LOO` is LOO field sensitivity: it measures dependence on existing
measurements and is not the canonical acquisition by itself.

The fitted latent-process kernel defines the support deficit. In normalized
coordinates,

\[
v_{support}(x)=k_f(x,x)-k_f(x,X)[K_f(X,X)+\Sigma]^{-1}k_f(X,x),
\]

\[
C(x)=\operatorname{clip}\left(\frac{v_{support}(x)}
{\max(k_f(x,x),\epsilon)},0,1\right).
\]

`WhiteKernel` is excluded from `k_f`; known observation variances, fitted
white noise, and the numerical GPR diagonal are included in `Sigma`. The
solve is performed without explicit matrix inversion, and nonfinite or badly
conditioned covariance calculations fail clearly.

The canonical field is exactly

\[
U_{KRISPU}(x)=S_{LOO}(x)\sqrt{C(x)}.
\]

The default recommender maximizes `krispu_uncertainty`. `posterior_std` is a
baseline diagnostic only. `raw_loo_sensitivity` is a diagnostic comparison
baseline; it is not the complete uncertainty field.
