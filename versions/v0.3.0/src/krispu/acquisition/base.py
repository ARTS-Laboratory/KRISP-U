"""Acquisition score protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Acquisition(Protocol):
    def __call__(self, diagnostics: object) -> NDArray[np.float64]: ...
