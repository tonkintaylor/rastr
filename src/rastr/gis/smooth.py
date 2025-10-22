"""Utilities for smoothing geometries.

Fork + Port of <https://github.com/philipschall/shapelysmooth> (Public domain)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from shapely.geometry import LineString, Polygon
from typing_extensions import assert_never

if TYPE_CHECKING:
    from numpy.typing import NDArray

T = TypeVar("T", bound=LineString | Polygon)


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
        return geometry.__class__(coords_smoothed)
    elif isinstance(geometry, Polygon):
        interior_coords_smoothed = [
            _catmull_rom(c, alpha=alpha, subdivs=subdivs) for c in interior_coords
        ]
        return geometry.__class__(coords_smoothed, holes=interior_coords_smoothed)
    else:
        assert_never(geometry)


def _catmull_rom(
    coords: NDArray,
    *,
    alpha: float = 0.5,
    subdivs: int = 8,
) -> list[tuple[float, float]]:
    arr = np.asarray(coords, dtype=float)
    n = arr.shape[0]
    if n < 2:
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

    # Shape of (segments, 4, D)
    segments = sliding_window_view(arr, (4, arr.shape[1]))[:, 0, :]

    # Distances and tangent values
    diffs = np.diff(segments, axis=1)
    dists = np.linalg.norm(diffs, axis=2)
    tangents = np.concatenate(
        [np.zeros((len(dists), 1)), np.cumsum(dists**alpha, axis=1)], axis=1
    )

    # Build ts per segment
    if subdivs > 1:
        seg_lens = (tangents[:, 2] - tangents[:, 1]) / subdivs
        u = np.linspace(1, subdivs - 1, subdivs - 1)
        ts = tangents[:, [1]] + seg_lens[:, None] * u  # (N-3, subdivs-1)
    else:
        ts = np.empty((len(segments), 0))

    # Vectorize over segments
    out_segments = []
    for seg, tang, tvals in zip(segments, tangents, ts, strict=False):
        if tvals.size:
            out_segments.append(_recursive_eval(seg, tang, tvals))
    if out_segments:
        all_midpoints = np.vstack(out_segments)
    else:
        all_midpoints = np.empty((0, arr.shape[1]))

    # Gather final output in order
    result = [tuple(arr[1])]
    idx = 0
    for k in range(len(segments)):
        block = all_midpoints[idx : idx + max(subdivs - 1, 0)]
        result.extend(map(tuple, block))
        result.append(tuple(segments[k, 2]))
        idx += max(subdivs - 1, 0)

    return result


def _recursive_eval(slice4: NDArray, tangents: NDArray, ts: NDArray) -> NDArray:
    """De Boor/De Casteljau-style recursive linear interpolation over 4 control points.

    Parameterized by the non-uniform 'tangents' values.
    """
    slice4 = np.asarray(slice4, dtype=float)
    tangents = np.asarray(tangents, dtype=float)
    ts = np.asarray(ts, dtype=float)
    bigm = ts.shape[0]
    bigd = slice4.shape[1]

    # Initialize points for all ts, shape (M, 4, D)
    points = np.broadcast_to(slice4, (bigm, 4, bigd)).copy()

    # Recursive interpolation, but vectorized across all ts
    for r in range(1, 4):
        idx = max(r - 2, 0)
        denom = tangents[r - idx : 4 - idx] - tangents[idx : 4 - r + idx]
        denom = np.where(denom == 0, np.nan, denom)  # avoid div 0

        # Compute weights for all parameter values at once
        left_w = (tangents[r - idx : 4 - idx][None, :] - ts[:, None]) / denom
        right_w = 1 - left_w

        # Weighted sums between consecutive points
        points = (
            left_w[..., None] * points[:, 0 : 4 - r, :]
            + right_w[..., None] * points[:, 1 : 5 - r, :]
        )

    # Result is first (and only) point at this level
    return points[:, 0, :]


def _get_coords(
    geometry: LineString | Polygon,
) -> tuple[NDArray, list[NDArray]]:
    if isinstance(geometry, LineString):
        return np.array(geometry.coords), []
    elif isinstance(geometry, Polygon):
        return np.array(geometry.exterior.coords), [
            np.array(hole.coords) for hole in geometry.interiors
        ]
    else:
        assert_never(geometry)
