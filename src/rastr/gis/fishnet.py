from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from shapely import box

from rastr.utils import ensure_pair

if TYPE_CHECKING:
    from geopandas.array import GeometryArray
    from numpy.typing import ArrayLike, NDArray


def create_point_grid(
    *, bounds: tuple[float, float, float, float], cell_size: tuple[float, float] | float
) -> tuple[NDArray, NDArray]:
    """Create a regular grid of point coordinates for raster centers.

    This function replicates the original grid generation logic that uses
    np.arange to ensure compatibility with existing code.

    Args:
        bounds: (xmin, ymin, xmax, ymax) bounding box.
        cell_size: Size of each grid cell as (width, height) or a single value for
            square cells.

    Returns:
        Tuple of (x_coords, y_coords) meshgrids for raster cell centers.
    """
    cell_size = ensure_pair(cell_size)
    x_width, y_height = cell_size

    xmin, ymin, xmax, ymax = bounds

    # Use the original logic with np.arange for exact compatibility
    x_coords = np.arange(xmin + x_width / 2, xmax + x_width / 2, x_width)
    y_coords = np.arange(ymax - y_height / 2, ymin - y_height / 2, -y_height)

    x_points, y_points = np.meshgrid(x_coords, y_coords)  # type: ignore[reportAssignmentType]
    return x_points, y_points


def get_point_grid_shape(
    *,
    bounds: tuple[float, float, float, float] | ArrayLike,
    cell_size: tuple[float, float] | float,
) -> tuple[int, int]:
    """Calculate the shape of the point grid based on bounds and cell size.

    Args:
        bounds: (xmin, ymin, xmax, ymax) bounding box.
        cell_size: Size of each grid cell as (width, height) or a single value for
            square cells.
    """
    cell_size = ensure_pair(cell_size)
    x_width, y_height = cell_size

    xmin, ymin, xmax, ymax = np.asarray(bounds)
    ncols_exact = (xmax - xmin) / x_width
    nrows_exact = (ymax - ymin) / y_height

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
    *, bounds: tuple[float, float, float, float], res: tuple[float, float] | float
) -> GeometryArray:
    """Generate a fishnet of polygons from bounds.

    The function generates a grid of polygons within the specified bounds, where each
    cell has dimensions defined by `res`. If the resolution does not perfectly divide
    the bounds' dimensions (i.e., if `res` is not a factor of (xmax - xmin) or
    (ymax - ymin)), the grid is still generated such that it fully covers the bounds.
    This can result in cells that extend beyond the specified bounds.

    Args:
        bounds: (xmin, ymin, xmax, ymax)
        res: Resolution as `(width, height)` or a single value for square cells.

    Returns:
        Shapely Polygons.
    """
    import geopandas as gpd

    res = ensure_pair(res)
    cell_width, cell_height = res

    # Use the shared helper function to create the point grid
    xx, yy = create_point_grid(bounds=bounds, cell_size=res)

    # Create rectangles centered on each grid point
    polygons = box(
        xx.ravel() - cell_width / 2,
        yy.ravel() - cell_height / 2,
        xx.ravel() + cell_width / 2,
        yy.ravel() + cell_height / 2,
    )

    # GeoSeries.array is typed as ExtensionArray in geopandas stubs, but at runtime
    # this is a GeometryArray for polygon geometries.
    return cast("GeometryArray", gpd.GeoSeries(polygons).array)
