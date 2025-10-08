from __future__ import annotations

import importlib.util
import warnings
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import rasterio.features
import rasterio.transform
from affine import Affine
from pyproj import CRS
from shapely.geometry import Point

from rastr.gis.fishnet import create_point_grid, get_point_grid_shape
from rastr.meta import RasterMeta
from rastr.raster import Raster

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    import geopandas as gpd
    from numpy.typing import ArrayLike
    from shapely.geometry import Polygon


TQDM_INSTALLED = importlib.util.find_spec("tqdm") is not None

_T = TypeVar("_T")


class MissingColumnsError(ValueError):
    """Raised when target columns are missing from the GeoDataFrame."""


class NonNumericColumnsError(ValueError):
    """Raised when target columns contain non-numeric data."""


class RasterizationError(ValueError):
    """Base exception for rasterization errors."""


class OverlappingGeometriesError(RasterizationError):
    """Raised when geometries overlap, which could lead to data loss."""


def raster_distance_from_polygon(
    polygon: Polygon,
    *,
    raster_meta: RasterMeta,
    extent_polygon: Polygon | None = None,
    snap_raster: Raster | None = None,
    show_pbar: bool = False,
) -> Raster:
    """Make a raster where each cell's value is its centre's distance to a polygon.

    The raster should use a projected coordinate system.

    Parameters:
        polygon: Polygon to measure distances to.
        raster_meta: Raster configuration (giving cell_size, CRS, etc.).
        extent_polygon: Polygon for raster cell extent; The bounding box of this
                        polygon is the bounding box of the output raster. Cells outside
                        this polygon but within the bounding box will be NaN-valued, and
                        cells will not be generated centred outside the bounding box of
                        this polygon.
        snap_raster: An alternative to using the extent_polygon. If provided, the raster
                     must have the exact same cell alignment as the snap_raster.
        show_pbar: Whether to show a progress bar during the distance calculation.

    Returns:
        Array storing the distance between cell centres and the polygon. Cell are
        NaN-valued if they are within the polygon or outside the extent polygon.

    Raises:
        ValueError: If the provided CRS is geographic (lat/lon).
    """
    if show_pbar and not TQDM_INSTALLED:
        msg = "The 'tqdm' package is not installed. Progress bars will not be shown."
        warnings.warn(msg, UserWarning, stacklevel=2)
        show_pbar = False

    # Check if the provided CRS is projected (cartesian)
    if raster_meta.crs.is_geographic:
        err_msg = (
            "The provided CRS is geographic (lat/lon). Please use a projected CRS."
        )
        raise ValueError(err_msg)

    if extent_polygon is None and snap_raster is None:
        err_msg = "Either 'extent_polygon' or 'snap_raster' must be provided. "
        raise ValueError(err_msg)
    elif extent_polygon is not None and snap_raster is not None:
        err_msg = "Only one of 'extent_polygon' or 'snap_raster' can be provided. "
        raise ValueError(err_msg)
    elif extent_polygon is None and snap_raster is not None:
        # Calculate the coordinates
        x, y = snap_raster.get_xy()

        # Create a mask to identify points for which distance should be calculated
        distance_extent = snap_raster.bbox.difference(polygon)
    elif extent_polygon is not None and snap_raster is None:
        x, y = create_point_grid(
            bounds=extent_polygon.bounds, cell_size=raster_meta.cell_size
        )
        distance_extent = extent_polygon.difference(polygon)
    else:
        raise AssertionError

    pts = [Point(x, y) for x, y in zip(x.flatten(), y.flatten(), strict=True)]

    _pts = _pbar(pts, desc="Finding points within extent") if show_pbar else pts
    mask = [distance_extent.intersects(pt) for pt in _pts]

    _pts = _pbar(pts, desc="Calculating distances") if show_pbar else pts
    distances = np.where(mask, np.array([polygon.distance(pt) for pt in _pts]), np.nan)
    distance_raster = distances.reshape(x.shape)

    return Raster(arr=distance_raster, raster_meta=raster_meta)


def _pbar(iterable: Iterable[_T], *, desc: str | None = None) -> Iterable[_T]:
    from tqdm import tqdm

    return tqdm(iterable, desc=desc)


def full_raster(
    raster_meta: RasterMeta,
    *,
    bounds: tuple[float, float, float, float],
    fill_value: float = np.nan,
) -> Raster:
    """Create a raster with a specified fill value for all cells."""
    shape = get_point_grid_shape(bounds=bounds, cell_size=raster_meta.cell_size)
    arr = np.full(shape, fill_value, dtype=np.float32)
    return Raster(arr=arr, raster_meta=raster_meta)


def rasterize_gdf(
    gdf: gpd.GeoDataFrame,
    *,
    raster_meta: RasterMeta,
    target_cols: Collection[str],
) -> list[Raster]:
    """Rasterize geometries from a GeoDataFrame.

    Supports polygons, points, linestrings, and other geometry types.
    Gaps will be set as NaN.

    Args:
        gdf: The geometries to rasterize (polygons, points, linestrings, etc.).
        raster_meta: Metadata for the created rasters.
        target_cols: A list of columns from the GeoDataFrame containing numeric
                     datatypes. Each column will correspond to a separate raster
                     in the output.

    Returns:
        Rasters for each column in `target_cols`.

    Raises:
        MissingColumnsError: If any of the target columns are not found in the
                             GeoDataFrame.
        NonNumericColumnsError: If any of the target columns contain non-numeric data.
        OverlappingGeometriesError: If any geometries overlap, which could lead to
                                    data loss in the rasterization process.
    """
    # Validate inputs using helper functions
    _validate_columns_exist(gdf, target_cols)
    _validate_columns_numeric(gdf, target_cols)
    _validate_no_overlapping_geometries(gdf)

    # Get the bounds from the GeoDataFrame and expand them to include potential gaps
    bounds = gdf.total_bounds
    min_x, min_y, max_x, max_y = bounds
    cell_size = raster_meta.cell_size

    # Expand bounds by at least one cell size to ensure there are potential gaps
    buffer = cell_size
    expanded_bounds = (min_x - buffer, min_y - buffer, max_x + buffer, max_y + buffer)

    # Create point grid to get raster dimensions and transform
    shape = get_point_grid_shape(bounds=expanded_bounds, cell_size=cell_size)

    # Create the affine transform for rasterization
    transform = Affine.translation(
        expanded_bounds[0], expanded_bounds[3]
    ) * Affine.scale(cell_size, -cell_size)

    # Create rasters for each target column using rasterio.features.rasterize
    rasters = []
    for col in target_cols:
        # Create (geometry, value) pairs for rasterization
        shapes = [
            (geom, value) for geom, value in zip(gdf.geometry, gdf[col], strict=True)
        ]

        # Rasterize the geometries with their values
        raster_array = rasterio.features.rasterize(
            shapes,
            out_shape=shape,
            transform=transform,
            # Fill gaps with NaN
            fill=np.nan,  # type: ignore[reportArgumentType] docstring contradicts inferred annotation
            dtype=np.float32,
        )

        # Create Raster
        raster = Raster(arr=raster_array, raster_meta=raster_meta)
        rasters.append(raster)

    return rasters


def _validate_columns_exist(
    gdf: gpd.GeoDataFrame, target_cols: Collection[str]
) -> None:
    """Validate that all target columns exist in the GeoDataFrame.

    Args:
        gdf: The GeoDataFrame to check.
        target_cols: List of column names to validate.

    Raises:
        MissingColumnsError: If any columns are missing.
    """
    missing_cols = [col for col in target_cols if col not in gdf.columns]
    if missing_cols:
        msg = f"Target columns not found in GeoDataFrame: {missing_cols}"
        raise MissingColumnsError(msg)


def _validate_columns_numeric(
    gdf: gpd.GeoDataFrame, target_cols: Collection[str]
) -> None:
    """Validate that all target columns contain numeric data.

    Args:
        gdf: The GeoDataFrame to check.
        target_cols: List of column names to validate.

    Raises:
        NonNumericColumnsError: If any columns contain non-numeric data.
    """
    import pandas as pd

    non_numeric_cols = []
    for col in target_cols:
        if not pd.api.types.is_numeric_dtype(gdf[col]):
            non_numeric_cols.append(col)
    if non_numeric_cols:
        msg = f"Target columns must contain numeric data: {non_numeric_cols}"
        raise NonNumericColumnsError(msg)


def _validate_no_overlapping_geometries(gdf: gpd.GeoDataFrame) -> None:
    """Validate that geometries do not overlap.

    Args:
        gdf: The GeoDataFrame to check for overlapping geometries.

    Raises:
        OverlappingGeometriesError: If any geometries overlap.
    """
    # Check for overlaps by testing each geometry against all others
    geometries = gdf.geometry.to_numpy()

    for i in range(len(geometries)):
        for j in range(i + 1, len(geometries)):
            geom_i = geometries[i]
            geom_j = geometries[j]

            # Skip invalid geometries
            if not geom_i.is_valid or not geom_j.is_valid:
                continue

            # Check if geometries overlap (not just touch)
            if geom_i.overlaps(geom_j):
                msg = (
                    f"Overlapping geometries detected at indices {i} and {j}. "
                    "Overlapping geometries can lead to data loss during rasterization."
                )
                raise OverlappingGeometriesError(msg)


def raster_from_point_cloud(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    crs: CRS | str,
    cell_size: float | None = None,
) -> Raster:
    """Create a raster from a point cloud via interpolation.

    Interpolation is only possible within the convex hull of the points. Outside of
    this, cells will be NaN-valued.

    All (x,y) points must be unique.

    Args:
        x: X coordinates of points.
        y: Y coordinates of points.
        z: Values at each (x, y) point to assign the raster.
        crs: Coordinate reference system for the (x, y) coordinates.
        cell_size: Desired cell size for the raster. If None, a heuristic is used based
                   on the spacing between (x, y) points.

    Returns:
        Raster containing the interpolated values.

    Raises:
        ValueError: If any (x, y) points are duplicated, or if they are all collinear.
    """
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import KDTree, QhullError

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    crs = CRS.from_user_input(crs)

    # Validate input arrays
    if len(x) != len(y) or len(x) != len(z):
        msg = "Length of x, y, and z must be equal."
        raise ValueError(msg)
    xy_finite_mask = np.isfinite(x) & np.isfinite(y)
    if np.any(~xy_finite_mask):
        msg = "Some (x,y) points are NaN-valued or non-finite. These will be ignored."
        warnings.warn(msg, stacklevel=2)
        x = x[xy_finite_mask]
        y = y[xy_finite_mask]
        z = z[xy_finite_mask]
    if len(x) < 3:
        msg = (
            "At least three valid (x, y, z) points are required to triangulate a "
            "surface."
        )
        raise ValueError(msg)
    # Check for duplicate (x, y) points
    xy_points = np.column_stack((x, y))
    if len(xy_points) != len(np.unique(xy_points, axis=0)):
        msg = "Duplicate (x, y) points found. Each (x, y) point must be unique."
        raise ValueError(msg)

    # Heuristic for cell size if not provided
    if cell_size is None:
        # Half the 5th percentile of nearest neighbor distances between the (x,y) points
        tree = KDTree(xy_points)
        distances, _ = tree.query(xy_points, k=2)
        distances: np.ndarray
        cell_size = float(np.percentile(distances[distances > 0], 5)) / 2

    # Compute bounds from data
    minx, miny, maxx, maxy = np.min(x), np.min(y), np.max(x), np.max(y)

    # Compute grid shape
    width = int(np.ceil((maxx - minx) / cell_size))
    height = int(np.ceil((maxy - miny) / cell_size))
    shape = (height, width)

    # Compute transform: upper left corner is (minx, maxy)
    transform = Affine.translation(minx, maxy) * Affine.scale(cell_size, -cell_size)

    # Create grid coordinates for raster cells
    rows, cols = np.indices(shape)
    xs, ys = rasterio.transform.xy(
        transform=transform, rows=rows, cols=cols, offset="center"
    )
    grid_x = np.array(xs).ravel()
    grid_y = np.array(ys).ravel()

    # Perform interpolation
    try:
        interpolator = LinearNDInterpolator(
            points=xy_points, values=z, fill_value=np.nan
        )
    except QhullError as err:
        msg = (
            "Failed to interpolate. This may be due to insufficient or "
            "degenerate input points. Ensure that the (x, y) points are not all "
            "collinear (i.e. that the convex hull is non-degenerate)."
        )
        raise ValueError(msg) from err

    grid_values = np.array(interpolator(np.column_stack((grid_x, grid_y))))

    arr = grid_values.reshape(shape).astype(np.float32)

    raster_meta = RasterMeta(
        cell_size=cell_size,
        crs=crs,
        transform=transform,
    )
    return Raster(arr=arr, raster_meta=raster_meta)
