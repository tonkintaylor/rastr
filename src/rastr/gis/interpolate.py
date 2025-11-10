from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


class InterpolationError(ValueError):
    """Exception for interpolation-related errors."""


def interpn_kernel(
    points: np.ndarray,
    values: np.ndarray,
    *,
    xi: np.ndarray,
    kernel: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Interpolate scattered data to new points, with optional kernel transformation.

    For example, you could provide a kernel to transform cartesian coordinate points
    to polar coordinates before interpolation, giving interpolation which follows the
    circular pattern of the data.

    Args:
        points: Array of shape (n_points, n_dimensions) representing the input points.
        values: Array of shape (n_points,) representing the values at each input point.
        xi: Array of shape (m_points, n_dimensions) representing the points to
            interpolate to.
        kernel: Optional function to transform points (and xi) before interpolation.
    """
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import QhullError

    if kernel is not None:
        xi = kernel(xi)
        points = kernel(points)
    try:
        interpolator = LinearNDInterpolator(
            points=points, values=values, fill_value=np.nan
        )
    except QhullError as err:
        msg = (
            "Failed to interpolate. This may be due to insufficient or "
            "degenerate input points. Ensure that the (x, y) points are not all "
            "collinear (i.e. that the convex hull is non-degenerate)."
        )
        raise InterpolationError(msg) from err

    grid_values = np.array(interpolator(xi))
    return grid_values
