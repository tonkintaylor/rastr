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

    if "south" in dir_x or "down" in dir_x:
        x_sign = -1
    elif "north" in dir_x or "up" in dir_x:
        x_sign = +1
    elif "west" in dir_x or "left" in dir_x:
        x_sign = -1
    elif "east" in dir_x or "right" in dir_x:
        x_sign = +1
    else:
        msg = (
            f"Could not determine x-axis direction from CRS axis info '{dir_x}'. "
            "Falling back to +1 (increasing to the right)."
        )
        warnings.warn(msg, stacklevel=2)
        x_sign = +1

    if "north" in dir_y or "up" in dir_y:
        y_sign = -1
    elif "south" in dir_y or "down" in dir_y:
        y_sign = +1
    elif "east" in dir_y or "right" in dir_y:
        y_sign = -1
    elif "west" in dir_y or "left" in dir_y:
        y_sign = +1
    else:
        msg = (
            f"Could not determine y-axis direction from CRS axis info '{dir_y}'. "
            "Falling back to -1 (increasing upwards)."
        )
        warnings.warn(msg, stacklevel=2)
        y_sign = -1

    return x_sign, y_sign
