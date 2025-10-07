from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely import BufferCapStyle, BufferJoinStyle

if TYPE_CHECKING:
    from geopandas.array import GeometryArray
    from numpy.typing import NDArray


def create_point_grid(
    *, bounds: tuple[float, float, float, float], cell_size: float
) -> tuple[NDArray, NDArray]:
    """Create a regular grid of point coordinates for raster centers.

    This function replicates the original grid generation logic that uses
    np.arange to ensure compatibility with existing code.

    Args:
        bounds: (xmin, ymin, xmax, ymax) bounding box.
        cell_size: Size of each grid cell.

    Returns:
        Tuple of (x_coords, y_coords) meshgrids for raster cell centers.
    """
    xmin, ymin, xmax, ymax = bounds

    # Use the original logic with np.arange for exact compatibility
    x_coords = np.arange(xmin + cell_size / 2, xmax + cell_size / 2, cell_size)
    y_coords = np.arange(ymax - cell_size / 2, ymin - cell_size / 2, -cell_size)

    x_points, y_points = np.meshgrid(x_coords, y_coords)  # type: ignore[reportAssignmentType]
    return x_points, y_points


def get_point_grid_shape(
    *, bounds: tuple[float, float, float, float], cell_size: float
) -> tuple[int, int]:
    """Calculate the shape of the point grid based on bounds and cell size."""

    xmin, ymin, xmax, ymax = bounds
    ncols_exact = (xmax - xmin) / cell_size
    nrows_exact = (ymax - ymin) / cell_size

    # Use round for values very close to integers to avoid floating-point
    # sensitivity while maintaining ceil behavior for truly fractional values
    if np.isclose(ncols_exact, np.round(ncols_exact)):
        ncols = int(np.round(ncols_exact))
    else:
        ncols = int(np.ceil(ncols_exact))

    if np.isclose(nrows_exact, np.round(nrows_exact)):
        nrows = int(np.round(nrows_exact))
    else:
        nrows = int(np.ceil(nrows_exact))

    return nrows, ncols


def create_fishnet(
    *, bounds: tuple[float, float, float, float], res: float
) -> GeometryArray:
    """Generate a fishnet of polygons from bounds.

    The function generates a grid of polygons within the specified bounds, where each
    cell has dimensions defined by `res`. If the resolution does not perfectly divide
    the bounds' dimensions (i.e., if `res` is not a factor of (xmax - xmin) or
    (ymax - ymin)), the grid is still generated such that it fully covers the bounds.
    This can result in cells that extend beyond the specified bounds.

    Args:
        bounds: (xmin, ymin, xmax, ymax)
        res: resolution (cell size)

    Returns:
        Shapely Polygons.
    """
    import geopandas as gpd

    # Use the shared helper function to create the point grid
    xx, yy = create_point_grid(bounds=bounds, cell_size=res)

    # Create points from the grid coordinates
    points = gpd.points_from_xy(xx.ravel(), yy.ravel())

    # Buffer the points to create square polygons
    polygons = points.buffer(
        res / 2, cap_style=BufferCapStyle.square, join_style=BufferJoinStyle.mitre
    )

    return polygons
