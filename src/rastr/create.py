from collections.abc import Iterable
from functools import partial

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
from affine import Affine
from shapely.geometry import Point, Polygon
from tqdm.notebook import tqdm

from rastr.gis.fishnet import create_point_grid, get_point_grid_shape
from rastr.meta import RasterMeta
from rastr.raster import RasterModel


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
    snap_raster: RasterModel | None = None,
    show_pbar: bool = False,
) -> RasterModel:
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
    if extent_polygon is None and snap_raster is None:
        err_msg = "Either 'extent_polygon' or 'snap_raster' must be provided. "
        raise ValueError(err_msg)
    elif extent_polygon is not None and snap_raster is not None:
        err_msg = "Only one of 'extent_polygon' or 'snap_raster' can be provided. "
        raise ValueError(err_msg)

    if not show_pbar:

        def _pbar(x: Iterable) -> None:
            return x  # No-op if no progress bar is needed

    # Check if the provided CRS is projected (cartesian)
    if raster_meta.crs.is_geographic:
        err_msg = (
            "The provided CRS is geographic (lat/lon). Please use a projected CRS."
        )
        raise ValueError(err_msg)

    # Calculate the coordinates
    if snap_raster is not None:
        x, y = snap_raster.get_xy()
    else:
        x, y = create_point_grid(
            bounds=extent_polygon.bounds, cell_size=raster_meta.cell_size
        )

    points = [Point(x, y) for x, y in zip(x.flatten(), y.flatten(), strict=True)]

    # Create a mask to identify points for which distance should be calculated
    if extent_polygon is not None:
        distance_extent = extent_polygon.difference(polygon)
    else:
        distance_extent = snap_raster.bbox.difference(polygon)

    if show_pbar:
        _pbar = partial(tqdm, desc="Finding points within extent")
    mask = [distance_extent.intersects(point) for point in _pbar(points)]

    if show_pbar:
        _pbar = partial(tqdm, desc="Calculating distances")
    distances = np.where(
        mask, np.array([polygon.distance(point) for point in _pbar(points)]), np.nan
    )
    distance_raster = distances.reshape(x.shape)

    return RasterModel(arr=distance_raster, raster_meta=raster_meta)


def full_raster(
    raster_meta: RasterMeta,
    *,
    bounds: tuple[float, float, float, float],
    fill_value: float = np.nan,
) -> RasterModel:
    """Create a raster with a specified fill value for all cells."""
    shape = get_point_grid_shape(bounds=bounds, cell_size=raster_meta.cell_size)
    arr = np.full(shape, fill_value, dtype=np.float32)
    return RasterModel(arr=arr, raster_meta=raster_meta)


def rasterize_gdf(
    gdf: gpd.GeoDataFrame,
    *,
    raster_meta: RasterMeta,
    target_cols: list[str],
) -> list[RasterModel]:
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
            fill=np.nan,  # Fill gaps with NaN
            dtype=np.float32,
        )

        # Create RasterModel
        raster = RasterModel(arr=raster_array, raster_meta=raster_meta)
        rasters.append(raster)

    return rasters


def _validate_columns_exist(gdf: gpd.GeoDataFrame, target_cols: list[str]) -> None:
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


def _validate_columns_numeric(gdf: gpd.GeoDataFrame, target_cols: list[str]) -> None:
    """Validate that all target columns contain numeric data.

    Args:
        gdf: The GeoDataFrame to check.
        target_cols: List of column names to validate.

    Raises:
        NonNumericColumnsError: If any columns contain non-numeric data.
    """
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
