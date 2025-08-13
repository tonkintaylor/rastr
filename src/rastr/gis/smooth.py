"""Utilities for smoothing geometries.

Fork + Port of <https://github.com/philipschall/shapelysmooth> (Public domain)
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from shapely.geometry import LineString, Polygon
from typing_extensions import assert_never

T: TypeAlias = LineString | Polygon


class InputeTypeError(TypeError):
    """Raised when the input geometry is of the incorrect type."""


def catmull_rom_smooth(geometry: T, alpha: float = 0.5, subdivs: int = 10) -> T:
    """Polyline smoothing using Catmull-Rom splines.

    Args:
        geometry: The geometry to smooth
        alpha: The tension parameter, between 0 and 1 inclusive. Defaults to 0.5.
               - For uniform Catmull-Rom splines, alpha = 0.
               - For centripetal Catmull-Rom splines, alpha = 0.5.
               - For chordal Catmull-Rom splines, alpha = 1.0.
        subdivs:
            Number of subdivisions of each polyline segment. Default value: 10.

    Returns: The smoothed geometry.
    """
    coords, interior_coords = _get_coords(geometry)
    coords_smoothed = _catmull_rom(coords, alpha=alpha, subdivs=subdivs)
    if isinstance(geometry, LineString):
        return type(geometry)(coords_smoothed)
    elif isinstance(geometry, Polygon):
        interior_coords_smoothed = [
            _catmull_rom(c, alpha=alpha, subdivs=subdivs) for c in interior_coords
        ]
        return type(geometry)(coords_smoothed, holes=interior_coords_smoothed)
    else:
        assert_never(geometry)


def _catmull_rom(
    coords: np.ndarray,
    *,
    alpha: float = 0.5,
    subdivs: int = 8,
) -> list[tuple[float, float]]:
    arr = np.asarray(coords, dtype=float)
    if arr.shape[0] < 2:
        return arr.tolist()

    is_closed = np.allclose(arr[0], arr[-1])
    if is_closed:
        arr = np.vstack([arr[-2], arr, arr[2]])
    else:
        arr = np.vstack(
            [
                2.0 * arr[0] + 1.0 * arr[1],
                arr,
                2.0 * arr[-1] + 0.0 * arr[-2],
            ]
        )

    new_ls = [tuple(arr[1])]
    for k in range(len(arr) - 3):
        slice4 = arr[k : k + 4]
        tangents = [0.0]
        for j in range(3):
            dist = float(np.linalg.norm(slice4[j + 1] - slice4[j]))
            tangents.append(float(tangents[-1] + dist**alpha))

        # Resample: subdivs-1 samples strictly between t1 and t2
        seg_len = (tangents[2] - tangents[1]) / float(subdivs)
        if subdivs > 1:
            ts = np.linspace(tangents[1] + seg_len, tangents[2] - seg_len, subdivs - 1)
        else:
            ts = np.array([])

        interpolants = _recursive_eval(slice4, tangents, ts)
        new_ls.extend(interpolants)
        new_ls.append(tuple(slice4[2]))
    return new_ls


def _recursive_eval(
    slice4: np.ndarray, tangents: list[float], ts: np.ndarray
) -> list[tuple[float, float]]:
    """De Boor/De Casteljau-style recursive linear interpolation over 4 control points.

    Parameterized by the non-uniform 'tangents' values.
    """
    # N.B. comments are LLM-generated

    out = []
    for tp in ts:
        # Start with the 4 control points for this segment
        points = slice4.copy()
        # Perform 3 levels of linear interpolation (De Casteljau's algorithm)
        for r in range(1, 4):
            idx = max(r - 2, 0)
            new_points = []
            # Interpolate between points at this level
            for i in range(4 - r):
                # Compute denominator for parameterization
                denom = tangents[i + r - idx] - tangents[i + idx]
                if denom == 0:
                    # If degenerate (coincident tangents), use midpoint
                    left_w = right_w = 0.5
                else:
                    # Otherwise, compute weights for linear interpolation
                    left_w = (tangents[i + r - idx] - tp) / denom
                    right_w = (tp - tangents[i + idx]) / denom
                # Weighted average of the two points
                pt = left_w * points[i] + right_w * points[i + 1]
                new_points.append(pt)
            # Move to the next level with the new set of points
            points = np.array(new_points)
        # The final point is the interpolated value for this parameter tp
        out.append(tuple(points[0]))
    return out


def _get_coords(
    geometry: LineString | Polygon,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if isinstance(geometry, LineString):
        return np.array(geometry.coords), []
    elif isinstance(geometry, Polygon):
        return np.array(geometry.exterior.coords), [
            np.array(hole.coords) for hole in geometry.interiors
        ]
    else:
        assert_never(geometry)
