from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def ensure_pair(value: T | tuple[T, T]) -> tuple[T, T]:
    """Ensure the value is a tuple of two elements.

    If a single value is provided, it will be duplicated to create a tuple.
    """
    if isinstance(value, tuple):
        if len(value) != 2:
            msg = "Tuple must have exactly 2 elements."
            raise ValueError(msg)
        return value
    else:
        return (value, value)
