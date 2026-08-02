"""Reserved analytic fixed-kernel LOO backend.

The first v0.3.0 pass intentionally keeps brute-force LOO as the only backend
so its field-level predictions are easy to audit.  The analytic formulas are
documented for the next pass and are not silently substituted here.
"""

from __future__ import annotations


def compute_analytic_loo(*args, **kwargs):
    raise NotImplementedError(
        "Analytic candidate-level LOO is deferred until equivalence tests against "
        "the brute-force reference backend are added."
    )
