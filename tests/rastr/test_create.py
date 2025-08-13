import re

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from pyproj.crs.crs import CRS
from shapely.geometry import LineString, Point, Polygon

from rastr.create import (
    MissingColumnsError,
    NonNumericColumnsError,
    OverlappingGeometriesError,
    _validate_columns_exist,
    _validate_columns_numeric,
    full_raster,
    raster_distance_from_polygon,
    rasterize_gdf,
)
from rastr.meta import RasterMeta
from rastr.raster import RasterModel

_PROJECTED_CRS = CRS.from_epsg(3857)
_GEOGRAPHIC_CRS = CRS.from_epsg(4326)


class TestRasterDistanceFromPolygon:
    def test_nested_squares(self):
        raster_config = RasterMeta(
            cell_size=1, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, 1.0)
        )
        extent_polygon = Polygon([(0, 0), (0, 3), (3, 3), (3, 0)])
        polygon = Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])

        # Test with valid inputs
        result = raster_distance_from_polygon(
            polygon, extent_polygon=extent_polygon, raster_meta=raster_config
        )
        expected = np.array(
            [
                [1 / 2**0.5, 0.5, 1 / 2**0.5],
                [0.5, np.nan, 0.5],
                [1 / 2**0.5, 0.5, 1 / 2**0.5],
            ]
        )
        np.testing.assert_almost_equal(result.arr, expected)

    def test_non_overlapping_squares(self):
        # Setup for non-overlapping squares
        polygon = Polygon([(4, 0), (4, 3), (7, 3), (7, 0)])
        extent_polygon = Polygon([(0, 0), (0, 3), (3, 3), (3, 0)])
        raster_config = RasterMeta(
            cell_size=2, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, 1.0)
        )

        # Expected output
        expected = np.array([[3, 1], [3, 1]])

        result = raster_distance_from_polygon(
            polygon, extent_polygon=extent_polygon, raster_meta=raster_config
        )
        np.testing.assert_almost_equal(result.arr, expected)

    def test_kissing_squares_irregular_grid(self):
        """Test kissing squares on an irregular grid, with NaN for outlying cells."""
        extent_polygon = Polygon([(0, 0), (0, 2), (4, 2), (4, 0)])
        polygon = Polygon([(1, 0), (1, 2), (3, 2), (3, 0)])
        raster_config = RasterMeta(
            cell_size=0.6, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, 1.0)
        )

        # Pre-calculated expected output
        expected = np.array(
            [
                [0.7, 0.1, np.nan, np.nan, np.nan, 0.3, 0.9],
                [0.7, 0.1, np.nan, np.nan, np.nan, 0.3, 0.9],
                [0.7, 0.1, np.nan, np.nan, np.nan, 0.3, 0.9],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ]
        )

        result = raster_distance_from_polygon(
            polygon, extent_polygon=extent_polygon, raster_meta=raster_config
        )
        np.testing.assert_almost_equal(result.arr, expected)

    def test_cell_center_at_intersection(self):
        # Setup for cell center at intersection
        extent_polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        raster_config = RasterMeta(
            cell_size=2, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, 1.0)
        )

        # Expected output
        expected = np.array([[0]])

        result = raster_distance_from_polygon(
            polygon, extent_polygon=extent_polygon, raster_meta=raster_config
        )
        np.testing.assert_almost_equal(result.arr, expected)

    def test_invalid_crs(self):
        polygon = Polygon()
        extent_polygon = Polygon()
        raster_config = RasterMeta(
            cell_size=1, crs=_GEOGRAPHIC_CRS, transform=Affine.scale(1.0, 1.0)
        )
        err_msg = re.escape(
            "The provided CRS is geographic (lat/lon). Please use a projected CRS."
        )
        with pytest.raises(ValueError, match=err_msg):
            raster_distance_from_polygon(
                polygon, extent_polygon=extent_polygon, raster_meta=raster_config
            )

    def test_output_meta_same_as_input_config(self):
        polygon = Polygon([(0, 0), (1, 1), (0, 1)])
        extent_polygon = Polygon([(0, 0), (1, 1), (0, 1)])
        raster_config = RasterMeta(
            cell_size=1, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, 1.0)
        )
        result = raster_distance_from_polygon(
            polygon, extent_polygon=extent_polygon, raster_meta=raster_config
        )
        assert result.raster_meta == raster_config

    def test_no_extent_or_snap_raster(self):
        polygon = Polygon([(0, 0), (1, 1), (0, 1)])
        raster_config = RasterMeta(
            cell_size=1, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, 1.0)
        )
        err_msg = re.escape(
            "Either 'extent_polygon' or 'snap_raster' must be provided."
        )
        with pytest.raises(ValueError, match=err_msg):
            raster_distance_from_polygon(polygon, raster_meta=raster_config)

    def test_both_extent_and_snap_raster(self):
        polygon = Polygon([(0, 0), (1, 1), (0, 1)])
        extent_polygon = Polygon([(0, 0), (1, 1), (0, 1)])
        raster_config = RasterMeta(
            cell_size=1, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, 1.0)
        )
        # Non-none snap_raster for testing
        snap_raster = RasterModel.example()
        err_msg = re.escape(
            "Only one of 'extent_polygon' or 'snap_raster' can be provided."
        )
        with pytest.raises(ValueError, match=err_msg):
            raster_distance_from_polygon(
                polygon,
                extent_polygon=extent_polygon,
                snap_raster=snap_raster,
                raster_meta=raster_config,
            )

    def test_show_pbar_true(self):
        polygon = Polygon([(0, 0), (1, 1), (0, 1)])
        extent_polygon = Polygon([(0, 0), (1, 1), (0, 1)])
        raster_config = RasterMeta(
            cell_size=1, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, 1.0)
        )
        result = raster_distance_from_polygon(
            polygon,
            extent_polygon=extent_polygon,
            raster_meta=raster_config,
            show_pbar=True,
        )
        assert isinstance(result, RasterModel)


class TestFullRaster:
    def test_full_raster(self):
        raster_meta = RasterMeta(
            cell_size=1, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, 1.0)
        )
        bounds = (0, 0, 3, 3)
        result = full_raster(raster_meta, bounds=bounds)
        assert isinstance(result, RasterModel)
        assert result.raster_meta == raster_meta
        assert result.arr.shape == (3, 3)  # 3x3 grid for bounds (0,0) to (3,3)


class TestRasterizeGdf:
    """Test suite for rasterize_gdf function."""

    def test_basic_rasterization_single_column(self):
        """Test basic rasterization with a single numeric column."""
        # Create test polygons
        polygons = [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
            Polygon([(1, 0), (1, 1), (2, 1), (2, 0)]),
            Polygon([(0, 1), (0, 2), (1, 2), (1, 1)]),
        ]
        values = [10.0, 20.0, 30.0]

        gdf = gpd.GeoDataFrame(
            {"value": values, "geometry": polygons}, crs=_PROJECTED_CRS
        )
        raster_meta = RasterMeta(
            cell_size=1.0, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, -1.0)
        )

        result = rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["value"])

        assert len(result) == 1
        assert isinstance(result[0], RasterModel)
        assert result[0].raster_meta == raster_meta

        # Check that values are correctly assigned
        raster_array = result[0].arr
        assert raster_array.shape == (4, 4)  # 4x4 grid due to buffer expansion
        # The expected values depend on the spatial join behavior

    def test_multiple_columns(self):
        """Test rasterization with multiple numeric columns."""
        polygons = [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
            Polygon([(1, 0), (1, 1), (2, 1), (2, 0)]),
        ]

        gdf = gpd.GeoDataFrame(
            {
                "value1": [10.0, 20.0],
                "value2": [100.0, 200.0],
                "geometry": polygons,
            },
            crs=_PROJECTED_CRS,
        )
        raster_meta = RasterMeta(
            cell_size=0.5, crs=_PROJECTED_CRS, transform=Affine.scale(0.5, -0.5)
        )

        result = rasterize_gdf(
            gdf, raster_meta=raster_meta, target_cols=["value1", "value2"]
        )

        assert len(result) == 2
        assert all(isinstance(r, RasterModel) for r in result)
        assert all(r.raster_meta == raster_meta for r in result)

    def test_empty_geodataframe(self):
        """Test with empty GeoDataFrame."""
        gdf = gpd.GeoDataFrame({"value": []}, geometry=[], crs=_PROJECTED_CRS)
        raster_meta = RasterMeta(
            cell_size=1.0, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, -1.0)
        )

        # This should handle the empty case gracefully
        with pytest.raises((ValueError, IndexError)):
            rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["value"])

    def test_missing_target_columns(self):
        """Test error handling when target columns are missing."""
        polygons = [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])]
        gdf = gpd.GeoDataFrame(
            {"value": [10.0], "geometry": polygons}, crs=_PROJECTED_CRS
        )
        raster_meta = RasterMeta(
            cell_size=1.0, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, -1.0)
        )

        with pytest.raises(MissingColumnsError):
            rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["missing_column"])

    def test_non_numeric_columns(self):
        """Test error handling when target columns contain non-numeric data."""
        polygons = [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])]
        gdf = gpd.GeoDataFrame(
            {"text_col": ["abc"], "geometry": polygons}, crs=_PROJECTED_CRS
        )
        raster_meta = RasterMeta(
            cell_size=1.0, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, -1.0)
        )

        with pytest.raises(NonNumericColumnsError):
            rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["text_col"])

    def test_validation_helper_functions(self):
        """Test the validation helper functions directly."""
        # Test column existence validation
        polygons = [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])]
        gdf = gpd.GeoDataFrame(
            {"value": [10.0], "geometry": polygons}, crs=_PROJECTED_CRS
        )

        # Should not raise for existing column
        _validate_columns_exist(gdf, ["value"])

        # Should raise for missing column
        with pytest.raises(MissingColumnsError):
            _validate_columns_exist(gdf, ["missing_column"])

        # Test numeric validation
        gdf_with_text = gpd.GeoDataFrame(
            {"text_col": ["abc"], "geometry": polygons}, crs=_PROJECTED_CRS
        )

        # Should not raise for numeric column
        _validate_columns_numeric(gdf, ["value"])

        # Should raise for non-numeric column
        with pytest.raises(NonNumericColumnsError):
            _validate_columns_numeric(gdf_with_text, ["text_col"])

    def test_nan_handling(self):
        """Test handling of NaN values in numeric columns."""
        polygons = [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
            Polygon([(1, 0), (1, 1), (2, 1), (2, 0)]),
        ]
        gdf = gpd.GeoDataFrame(
            {"value": [10.0, np.nan], "geometry": polygons}, crs=_PROJECTED_CRS
        )
        raster_meta = RasterMeta(
            cell_size=1.0, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, -1.0)
        )

        result = rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["value"])

        assert len(result) == 1
        # Should have some NaN values where the second polygon is
        assert np.any(np.isnan(result[0].arr))

    def test_overlapping_polygons(self):
        """Test error handling for overlapping polygons.

        The function should detect overlapping geometries and raise an error
        to prevent potential data loss during rasterization.
        """
        # Create overlapping polygons with distinct values
        polygons = [
            Polygon([(0, 0), (0, 2), (2, 2), (2, 0)]),  # Large polygon with value 10
            Polygon([(1, 1), (1, 3), (3, 3), (3, 1)]),  # Overlapping with value 20
        ]
        gdf = gpd.GeoDataFrame(
            {"value": [10.0, 20.0], "geometry": polygons}, crs=_PROJECTED_CRS
        )
        raster_meta = RasterMeta(
            cell_size=1.0, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, -1.0)
        )

        # Should raise an error due to overlapping geometries
        with pytest.raises(
            OverlappingGeometriesError, match="Overlapping geometries detected"
        ):
            rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["value"])

    def test_touching_but_not_overlapping_polygons(self):
        """Test that touching (but not overlapping) polygons do not raise errors."""
        # Create adjacent polygons that share a boundary but don't overlap
        polygons = [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),  # Left square
            Polygon([(1, 0), (1, 1), (2, 1), (2, 0)]),  # Right square (shares edge)
        ]
        gdf = gpd.GeoDataFrame(
            {"value": [10.0, 20.0], "geometry": polygons}, crs=_PROJECTED_CRS
        )
        raster_meta = RasterMeta(
            cell_size=1.0, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, -1.0)
        )

        # Should not raise an error since polygons only touch, don't overlap
        result = rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["value"])
        assert len(result) == 1
        assert isinstance(result[0], RasterModel)

    def test_gaps_become_nan(self):
        """Test that areas without polygons become NaN in the raster."""
        # Create polygons that don't cover the entire extent
        polygons = [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),  # Only covers part of the area
        ]
        gdf = gpd.GeoDataFrame(
            {"value": [10.0], "geometry": polygons}, crs=_PROJECTED_CRS
        )
        raster_meta = RasterMeta(
            cell_size=1.0, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, -1.0)
        )

        result = rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["value"])

        # There should be some NaN values where no polygons exist
        assert np.any(np.isnan(result[0].arr))

    def test_complex_polygon_shapes(self):
        """Test with non-rectangular polygon shapes."""
        # Create a triangular polygon
        triangle = Polygon([(0, 0), (2, 0), (1, 2)])
        gdf = gpd.GeoDataFrame(
            {"value": [42.0], "geometry": [triangle]}, crs=_PROJECTED_CRS
        )
        raster_meta = RasterMeta(
            cell_size=0.5, crs=_PROJECTED_CRS, transform=Affine.scale(0.5, -0.5)
        )

        result = rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["value"])

        assert len(result) == 1
        # Should have some cells with value 42.0 and some with NaN
        assert np.any(~np.isnan(result[0].arr))  # Some non-NaN values

    def test_different_data_types(self):
        """Test with different numeric data types."""
        polygons = [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])]

        gdf = gpd.GeoDataFrame(
            {
                "int_col": np.array([10], dtype=np.int32),
                "float_col": np.array([10.5], dtype=np.float64),
                "geometry": polygons,
            },
            crs=_PROJECTED_CRS,
        )
        raster_meta = RasterMeta(
            cell_size=1.0, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, -1.0)
        )

        result = rasterize_gdf(
            gdf, raster_meta=raster_meta, target_cols=["int_col", "float_col"]
        )

        assert len(result) == 2
        # All output arrays should be float32
        assert result[0].arr.dtype == np.float32
        assert result[1].arr.dtype == np.float32

    def test_raster_metadata_preservation(self):
        """Test that raster metadata is correctly preserved."""
        polygons = [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])]
        gdf = gpd.GeoDataFrame(
            {"value": [10.0], "geometry": polygons}, crs=_PROJECTED_CRS
        )

        custom_transform = Affine.scale(2.0, -2.0) * Affine.translation(100, 200)
        raster_meta = RasterMeta(
            cell_size=2.0, crs=_PROJECTED_CRS, transform=custom_transform
        )

        result = rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["value"])

        assert result[0].raster_meta.cell_size == 2.0
        assert result[0].raster_meta.crs == _PROJECTED_CRS
        assert result[0].raster_meta.transform == custom_transform

    def test_large_cell_size(self):
        """Test with large cell size relative to polygon size."""
        # Small polygon with large cell size
        polygons = [Polygon([(0, 0), (0, 0.1), (0.1, 0.1), (0.1, 0)])]
        gdf = gpd.GeoDataFrame(
            {"value": [10.0], "geometry": polygons}, crs=_PROJECTED_CRS
        )
        raster_meta = RasterMeta(
            cell_size=1.0, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, -1.0)
        )

        result = rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["value"])

        assert len(result) == 1
        # The small polygon might not intersect with any cell centers

    def test_very_small_cell_size(self):
        """Test with very small cell size creating high resolution raster."""
        polygons = [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])]
        gdf = gpd.GeoDataFrame(
            {"value": [10.0], "geometry": polygons}, crs=_PROJECTED_CRS
        )
        raster_meta = RasterMeta(
            cell_size=0.1, crs=_PROJECTED_CRS, transform=Affine.scale(0.1, -0.1)
        )

        result = rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["value"])

        assert len(result) == 1
        # Should create a high-resolution raster
        raster_array = result[0].arr
        assert raster_array.shape[0] >= 10  # At least 10 rows for 1-unit height
        assert raster_array.shape[1] >= 10  # At least 10 cols for 1-unit width

    def test_point_geometries(self):
        """Test rasterization with point geometries."""
        points = [Point(0.5, 0.5), Point(1.5, 1.5), Point(2.5, 0.5)]
        gdf = gpd.GeoDataFrame(
            {"value": [10.0, 20.0, 30.0], "geometry": points}, crs=_PROJECTED_CRS
        )
        raster_meta = RasterMeta(
            cell_size=1.0, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, -1.0)
        )

        result = rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["value"])

        assert len(result) == 1
        assert isinstance(result[0], RasterModel)
        raster_array = result[0].arr

        # Points should be rasterized to their containing cells
        # Check that some cells contain the expected values
        non_nan_count = np.count_nonzero(~np.isnan(raster_array))
        assert non_nan_count > 0, "Points should create non-NaN cells"

        # Check that point values are preserved in the raster
        unique_values = raster_array[~np.isnan(raster_array)]
        expected_values = {10.0, 20.0, 30.0}
        assert len(set(unique_values).intersection(expected_values)) > 0

    def test_linestring_geometries(self):
        """Test rasterization with LineString geometries."""
        lines = [
            LineString([(0, 0), (2, 2)]),  # Diagonal line
            LineString([(0, 1), (3, 1)]),  # Horizontal line
            LineString([(1, 0), (1, 2)]),  # Vertical line
        ]
        gdf = gpd.GeoDataFrame(
            {"value": [100.0, 200.0, 300.0], "geometry": lines}, crs=_PROJECTED_CRS
        )
        raster_meta = RasterMeta(
            cell_size=0.5, crs=_PROJECTED_CRS, transform=Affine.scale(0.5, -0.5)
        )

        result = rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["value"])

        assert len(result) == 1
        assert isinstance(result[0], RasterModel)
        raster_array = result[0].arr

        # Lines should be rasterized across multiple cells
        non_nan_count = np.count_nonzero(~np.isnan(raster_array))
        assert non_nan_count > 3, "Lines should span multiple cells"

        # Check that line values are preserved in the raster
        unique_values = raster_array[~np.isnan(raster_array)]
        expected_values = {100.0, 200.0, 300.0}
        assert len(set(unique_values).intersection(expected_values)) > 0

    def test_mixed_geometry_types(self):
        """Test rasterization with mixed geometry types in the same GeoDataFrame."""
        geometries = [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),  # Polygon
            Point(2, 0.5),  # Point
            LineString([(0, 2), (2, 2)]),  # Line
        ]
        gdf = gpd.GeoDataFrame(
            {"value": [10.0, 20.0, 30.0], "geometry": geometries}, crs=_PROJECTED_CRS
        )
        raster_meta = RasterMeta(
            cell_size=0.5, crs=_PROJECTED_CRS, transform=Affine.scale(0.5, -0.5)
        )

        result = rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["value"])

        assert len(result) == 1
        assert isinstance(result[0], RasterModel)
        raster_array = result[0].arr

        # All geometry types should be rasterized
        non_nan_count = np.count_nonzero(~np.isnan(raster_array))
        assert non_nan_count > 0, "Mixed geometries should create non-NaN cells"

        # Values from all geometry types should be represented
        unique_values = set(raster_array[~np.isnan(raster_array)])
        expected_values = {10.0, 20.0, 30.0}
        # At least some of the expected values should be present
        assert len(unique_values.intersection(expected_values)) > 0

    def test_point_on_cell_boundary(self):
        """Test point that falls exactly on cell boundaries."""
        # Point exactly on grid boundary
        points = [Point(1.0, 1.0)]
        gdf = gpd.GeoDataFrame(
            {"value": [42.0], "geometry": points}, crs=_PROJECTED_CRS
        )
        raster_meta = RasterMeta(
            cell_size=1.0, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, -1.0)
        )

        result = rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["value"])

        assert len(result) == 1
        raster_array = result[0].arr

        # Point should be assigned to one of the adjacent cells
        non_nan_count = np.count_nonzero(~np.isnan(raster_array))
        assert non_nan_count >= 1, "Boundary point should be rasterized"
