from __future__ import annotations

import importlib.util
import warnings
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
import rasterio.features
import rasterio.transform
from affine import Affine
from pyproj import CRS
from shapely.geometry import Point
from typing_extensions import assert_never

from rastr.gis.crs import get_affine_sign
from rastr.gis.fishnet import create_point_grid, get_point_grid_shape
from rastr.gis.interpolate import InterpolationError, interpn_kernel
from rastr.meta import RasterMeta
from rastr.raster import Raster, RasterModel

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    import geopandas as gpd
    from numpy.typing import ArrayLike, NDArray
    from shapely.geometry import Polygon
    from shapely.geometry.base import BaseGeometry


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
    xs, ys = get_affine_sign(raster_meta.crs)
    transform = Affine.translation(
        expanded_bounds[0], expanded_bounds[3]
    ) * Affine.scale(xs * cell_size, ys * cell_size)

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


def rasterize_z_gdf(
    gdf: gpd.GeoDataFrame,
    *,
    raster_meta: RasterMeta,
    agg: Literal["mean", "min", "max"] = "mean",
) -> RasterModel:
    """Rasterize interpolated Z-values from geometries in a GeoDataFrame.

    Handles overlapping geometries by aggregating values using a specified method.
    All geometries must be 3D (have Z coordinates) for interpolation to work.

    The Z-value for each cell is interpolated at the cell center.

    Args:
        gdf: GeoDataFrame containing 3D geometries with Z coordinates.
        raster_meta: Raster metadata (giving cell size, etc.)
        agg: Aggregation function to use for overlapping values ("mean", "min", "max").

    Returns:
        A raster of interpolated Z values.

    Raises:
        ValueError: If any geometries are not 3D.
    """
    gdf = gdf.copy()

    # Handle empty GeoDataFrame early
    if len(gdf) == 0:
        # Return a minimal raster with NaN values
        # Use a default 1x1 cell array since there's no geometry to determine bounds
        arr = np.full((1, 1), np.nan, dtype=np.float32)
        return RasterModel(arr=arr, raster_meta=raster_meta)

    _validate_geometries_are_3d(gdf)

    # Use the provided raster metadata instead of creating new bounds
    # This ensures the raster respects the requested grid alignment
    cell_size = raster_meta.cell_size
    transform = raster_meta.transform

    # Determine the bounds that would encompass the geometry while respecting the grid
    gdf_bounds = gdf.total_bounds
    minx_geom, miny_geom, maxx_geom, maxy_geom = gdf_bounds

    # Use geometry bounds directly without extra buffering
    geom_bounds = (minx_geom, miny_geom, maxx_geom, maxy_geom)

    # Calculate the grid shape needed to cover the geometry bounds
    shape = get_point_grid_shape(bounds=geom_bounds, cell_size=cell_size)
    height, width = shape

    # Create a new transform that aligns with the provided grid but covers the geometry
    # Start from the top-left corner that would align with the provided grid
    grid_origin_x = transform.c  # x-offset from original transform
    grid_origin_y = transform.f  # y-offset from original transform

    # Find the top-left corner of our raster that aligns with the original grid
    # and covers the geometry bounds
    cols_to_start = int((geom_bounds[0] - grid_origin_x) / cell_size)
    rows_to_start = int((grid_origin_y - geom_bounds[3]) / cell_size)

    # Calculate the actual bounds of our aligned raster
    actual_minx = grid_origin_x + cols_to_start * cell_size
    actual_maxy = grid_origin_y - rows_to_start * cell_size

    # Create the aligned transform
    aligned_transform = Affine.translation(actual_minx, actual_maxy) * Affine.scale(
        cell_size, -cell_size
    )

    # Update the raster metadata to use the aligned transform
    aligned_raster_meta = RasterMeta(
        cell_size=cell_size, crs=raster_meta.crs, transform=aligned_transform
    )

    # Generate grid coordinates for interpolation
    rows, cols = np.indices(shape)
    xs, ys = rasterio.transform.xy(aligned_transform, rows, cols, offset="center")
    x_coords = np.array(xs).ravel()
    y_coords = np.array(ys).ravel()

    # Create 2D accumulation arrays
    z_stack = []
    for geom in gdf.geometry:
        z_vals = _interpolate_z_in_geometry(geom, x_coords, y_coords)
        z_stack.append(z_vals)

    if not z_stack:
        z_raster = np.full((height, width), np.nan, dtype=np.float32)
        return RasterModel(arr=z_raster, raster_meta=aligned_raster_meta)

    z_stack = np.array(z_stack)  # Shape: (N, height * width)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="All-NaN slice encountered"
        )
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="Mean of empty slice"
        )

        if agg == "mean":
            z_agg = np.nanmean(z_stack, axis=0)
        elif agg == "min":
            z_agg = np.nanmin(z_stack, axis=0)
        elif agg == "max":
            z_agg = np.nanmax(z_stack, axis=0)
        else:
            assert_never(agg)

    z_raster = np.asarray(z_agg).reshape((height, width)).astype(np.float32)

    return RasterModel(arr=z_raster, raster_meta=aligned_raster_meta)


def _validate_geometries_are_3d(gdf: gpd.GeoDataFrame) -> None:
    """Validate that all geometries have 3D coordinates (Z values).

    Args:
        gdf: The GeoDataFrame to check for 3D geometries.

    Raises:
        ValueError: If any geometries are not 3D.
    """
    for idx, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue

        # Check if geometry has Z coordinates
        if not geom.has_z:
            msg = (
                f"Geometry at index {idx} is not 3D. Z-coordinates are required since "
                "they give the cell values during rasterization."
            )
            raise ValueError(msg)


def _interpolate_z_in_geometry(
    geometry: BaseGeometry, x: NDArray, y: NDArray
) -> NDArray[np.float64]:
    """Vectorized interpolation of Z values in a geometry at multiple (x, y) points.

    Parameters:
        geometry: Shapely geometry with Z coordinates (Polygon, LineString, etc.).
        x: Array of X coordinates, shape (N,).
        y: Array of Y coordinates, shape (N,).

    Returns:
        Array of interpolated Z values (NaN if outside convex hull or no boundary).
    """
    if x.shape != y.shape:
        msg = "x and y must have the same shape"
        raise ValueError(msg)

    # Extract coordinates from geometry
    coords = np.array(geometry.boundary.coords)
    if coords is None:
        return np.full_like(x, np.nan, dtype=np.float64)

    # Validate geometry for interpolation
    if not _is_valid_for_interpolation(geometry, coords):
        return np.full_like(x, np.nan, dtype=np.float64)

    xy_boundary = coords[:, :2]
    z_boundary = coords[:, 2]

    try:
        return interpn_kernel(xy_boundary, z_boundary, xi=np.column_stack((x, y)))
    except InterpolationError:
        return np.full_like(x, np.nan, dtype=np.float64)


def _is_valid_for_interpolation(geometry: BaseGeometry, coords: np.ndarray) -> bool:
    """Check if geometry is valid for interpolation."""
    # Need at least 3 points for interpolation (except for Point geometries)
    if len(coords) < 3 and geometry.geom_type != "Point":
        return False

    # For Point geometries, we can't interpolate
    return geometry.geom_type != "Point"


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
    crs = CRS.from_user_input(crs)
    x, y, z = _validate_xyz(
        np.asarray(x).ravel(), np.asarray(y).ravel(), np.asarray(z).ravel()
    )

    raster_meta, shape = RasterMeta.infer(x, y, cell_size=cell_size, crs=crs)
    arr = interpn_kernel(
        points=np.column_stack((x, y)),
        values=z,
        xi=np.column_stack(_get_grid(raster_meta, shape=shape)),
    ).reshape(shape)

    # We only support float rasters for now; we should preserve the input dtype if
    # possible
    if z.dtype in (np.float16, np.float32, np.float64):
        arr = arr.astype(z.dtype)
    else:
        arr = arr.astype(np.float64)

    return Raster(arr=arr, raster_meta=raster_meta)


def _validate_xyz(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    return x, y, z


def _get_grid(
    raster_meta: RasterMeta, *, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Get coordinates for raster cell centres based on raster metadata and shape."""
    rows, cols = np.indices(shape)
    xs, ys = rasterio.transform.xy(
        transform=raster_meta.transform, rows=rows, cols=cols, offset="center"
    )
    grid_x = np.array(xs).ravel()
    grid_y = np.array(ys).ravel()

    return grid_x, grid_y
