from __future__ import annotations

import warnings
from typing import Literal

from pyproj import CRS


def get_affine_sign(crs: CRS | str) -> tuple[Literal[+1, -1], Literal[+1, -1]]:
    """Return (x_sign, y_sign) for an Affine scale, given a CRS.

    Some coordinate systems may use unconventional axis directions, in which case
    the correct direction may not be possible to infer correctly. In these cases,
    the assumption is that x increases to the right, and y increases upwards.
    """
    crs = CRS.from_user_input(crs)

    # Try to detect horizontal axis directions from CRS metadata
    dir_x, dir_y, *_ = [(a.direction or "").lower() for a in crs.axis_info]

    try:
        if _is_conventional_direction(dir_x):
            x_sign = +1
        else:
            x_sign = -1
    except NotImplementedError:
        msg = (
            f"Could not determine x-axis direction from CRS axis info '{dir_x}'. "
            "Falling back to +1 (increasing to the right)."
        )
        warnings.warn(msg, stacklevel=2)
        x_sign = +1

    try:
        if _is_conventional_direction(dir_y):
            y_sign = -1
        else:
            y_sign = +1
    except NotImplementedError:
        msg = (
            f"Could not determine y-axis direction from CRS axis info '{dir_y}'. "
            "Falling back to -1 (increasing upwards)."
        )
        warnings.warn(msg, stacklevel=2)
        y_sign = -1

    return x_sign, y_sign


def _is_conventional_direction(direction: str) -> bool:
    """Return True if the axis direction indicates positive increase."""
    if (
        "north" in direction
        or "up" in direction
        or "east" in direction
        or "right" in direction
    ):
        return True
    elif (
        "south" in direction
        or "down" in direction
        or "west" in direction
        or "left" in direction
    ):
        return False
    else:
        raise NotImplementedError
