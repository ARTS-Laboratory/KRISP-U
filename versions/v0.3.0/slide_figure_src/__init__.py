"""Reproducible figures for the KRISP-U v0.3.0 presentation."""

from typing import Any


def generate_all(*args: Any, **kwargs: Any) -> Any:
    """Lazily import the batch generator so ``python -m`` stays warning-free."""

    from .generate_all import generate_all as _generate_all

    return _generate_all(*args, **kwargs)


__all__ = ["generate_all"]
