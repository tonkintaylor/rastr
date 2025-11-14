from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest
from affine import Affine
from pyproj.crs.crs import CRS
from shapely import (
    GeometryCollection,
    LinearRing,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
)
from shapely.geometry import LineString, Point, Polygon

from rastr.create import (
    MissingColumnsError,
    NonNumericColumnsError,
    OverlappingGeometriesError,
    _extract_coords,
    _interpolate_z_in_geometry,
    _validate_columns_exist,
    _validate_columns_numeric,
    full_raster,
    raster_distance_from_polygon,
    raster_from_contours,
    raster_from_point_cloud,
    rasterize_gdf,
    rasterize_z_gdf,
)
from rastr.meta import RasterMeta
from rastr.raster import Raster

if TYPE_CHECKING:
    from pathlib import Path

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
        snap_raster = Raster.example()
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
        assert isinstance(result, Raster)

    def test_show_pbar_without_tqdm_warns(self, monkeypatch: pytest.MonkeyPatch):
        # Arrange
        monkeypatch.setattr("rastr.create.TQDM_INSTALLED", False)

        polygon = Polygon([(0, 0), (1, 1), (0, 1)])
        extent_polygon = Polygon([(0, 0), (1, 1), (0, 1)])
        raster_config = RasterMeta(
            cell_size=1, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, 1.0)
        )

        # Act, Assert
        expected_msg = (
            "The 'tqdm' package is not installed. Progress bars will not be shown."
        )
        with pytest.warns(UserWarning, match=expected_msg):
            result = raster_distance_from_polygon(
                polygon,
                extent_polygon=extent_polygon,
                raster_meta=raster_config,
                show_pbar=True,
            )

        assert isinstance(result, Raster)


class TestFullRaster:
    def test_full_raster(self):
        raster_meta = RasterMeta(
            cell_size=1, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, 1.0)
        )
        bounds = (0, 0, 3, 3)
        result = full_raster(raster_meta, bounds=bounds)
        assert isinstance(result, Raster)
        assert result.raster_meta == raster_meta
        assert result.arr.shape == (3, 3)  # 3x3 grid for bounds (0,0) to (3,3)

    def test_full_raster_roundtrip_shape(self):
        """Test that full_raster(r.meta, bounds=r.bounds).shape == r.shape."""
        # Create a raster
        transform = Affine.translation(0, 3) * Affine.scale(1.0, -1.0)
        raster_meta = RasterMeta(cell_size=1.0, crs=_PROJECTED_CRS, transform=transform)
        arr = np.ones((3, 3))
        r1 = Raster(arr=arr, raster_meta=raster_meta)

        # Recreate using full_raster with the same bounds
        r2 = full_raster(r1.raster_meta, bounds=r1.bounds)

        assert r2.shape == r1.shape

    def test_full_raster_floating_point_robustness(self):
        """Test that full_raster handles floating-point errors in bounds."""
        raster_meta = RasterMeta(
            cell_size=1.0, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, 1.0)
        )

        # Bounds with tiny floating-point error (simulating computational error)
        bounds_with_fp_error = (0.0, 0.0, 3.0 + 1e-10, 3.0 + 1e-10)
        result = full_raster(raster_meta, bounds=bounds_with_fp_error)

        # Should still produce a 3x3 raster, not 4x4
        assert result.arr.shape == (3, 3)


class TestRasterizeGDF:
    def test_basic_rasterization_single_column(self):
        """Test basic rasterization with a single numeric column."""
        import geopandas as gpd

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
        assert isinstance(result[0], Raster)
        assert result[0].raster_meta == raster_meta

        # Check that values are correctly assigned
        raster_array = result[0].arr
        assert raster_array.shape == (4, 4)  # 4x4 grid due to buffer expansion
        # The expected values depend on the spatial join behavior

    def test_multiple_columns(self):
        """Test rasterization with multiple numeric columns."""
        import geopandas as gpd

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
        assert all(isinstance(r, Raster) for r in result)
        assert all(r.raster_meta == raster_meta for r in result)

    def test_empty_geodataframe(self):
        """Test with empty GeoDataFrame."""
        import geopandas as gpd

        gdf = gpd.GeoDataFrame({"value": []}, geometry=[], crs=_PROJECTED_CRS)
        raster_meta = RasterMeta(
            cell_size=1.0, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, -1.0)
        )

        # This should handle the empty case gracefully
        with pytest.raises((ValueError, IndexError)):
            rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["value"])

    def test_missing_target_columns(self):
        """Test error handling when target columns are missing."""
        import geopandas as gpd

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
        import geopandas as gpd

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
        import geopandas as gpd

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
        import geopandas as gpd

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
        import geopandas as gpd

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
        import geopandas as gpd

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
        assert isinstance(result[0], Raster)

    def test_gaps_become_nan(self):
        """Test that areas without polygons become NaN in the raster."""
        import geopandas as gpd

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
        import geopandas as gpd

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
        import geopandas as gpd

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
        import geopandas as gpd

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
        import geopandas as gpd

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
        import geopandas as gpd

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
        import geopandas as gpd

        points = [Point(0.5, 0.5), Point(1.5, 1.5), Point(2.5, 0.5)]
        gdf = gpd.GeoDataFrame(
            {"value": [10.0, 20.0, 30.0], "geometry": points}, crs=_PROJECTED_CRS
        )
        raster_meta = RasterMeta(
            cell_size=1.0, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, -1.0)
        )

        result = rasterize_gdf(gdf, raster_meta=raster_meta, target_cols=["value"])

        assert len(result) == 1
        assert isinstance(result[0], Raster)
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
        import geopandas as gpd

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
        assert isinstance(result[0], Raster)
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
        import geopandas as gpd

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
        assert isinstance(result[0], Raster)
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
        import geopandas as gpd

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

    def test_target_cols_as_tuple(self):
        """Test that target_cols accepts a tuple (Collection) instead of just list."""
        import geopandas as gpd

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

        # Use tuple instead of list for target_cols
        result = rasterize_gdf(
            gdf, raster_meta=raster_meta, target_cols=("value1", "value2")
        )

        assert len(result) == 2
        assert all(isinstance(r, Raster) for r in result)

    def test_target_cols_as_set(self):
        """Test that target_cols accepts a set (Collection) instead of just list."""
        import geopandas as gpd

        polygons = [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
        ]
        gdf = gpd.GeoDataFrame(
            {"value": [10.0], "geometry": polygons}, crs=_PROJECTED_CRS
        )
        raster_meta = RasterMeta(
            cell_size=1.0, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, -1.0)
        )

        # Use set instead of list for target_cols
        result = rasterize_gdf(gdf, raster_meta=raster_meta, target_cols={"value"})

        assert len(result) == 1
        assert isinstance(result[0], Raster)


class TestInterpolateZInGeometry:
    """Test suite for _interpolate_z_in_geometry function."""

    def test_basic_interpolation_triangle(self):
        """Test basic interpolation within a triangular polygon."""
        # Create a triangle with known Z values at vertices
        coords = np.array(
            [
                [0.0, 0.0, 10.0],  # Bottom-left: Z=10
                [2.0, 0.0, 20.0],  # Bottom-right: Z=20
                [1.0, 2.0, 30.0],  # Top: Z=30
                [0.0, 0.0, 10.0],  # Close the polygon
            ]
        )
        polygon = Polygon(coords[:, :2])
        polygon = polygon.__class__(coords)  # Create 3D polygon

        # Test point at centroid should get interpolated value
        x = np.array([1.0])
        y = np.array([0.67])  # Approximately 2/3 up the triangle

        result = _interpolate_z_in_geometry(polygon, x, y)

        assert len(result) == 1
        assert not np.isnan(result[0])
        # The interpolated value should be between min and max Z values
        assert 10.0 <= result[0] <= 30.0

    def test_interpolation_at_vertices(self):
        """Test interpolation at polygon vertices returns exact Z values."""
        coords = np.array(
            [
                [0.0, 0.0, 100.0],
                [1.0, 0.0, 200.0],
                [1.0, 1.0, 300.0],
                [0.0, 1.0, 400.0],
                [0.0, 0.0, 100.0],
            ]
        )
        polygon = Polygon(coords[:, :2])
        polygon = polygon.__class__(coords)

        # Test at each vertex
        x = np.array([0.0, 1.0, 1.0, 0.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        expected_z = np.array([100.0, 200.0, 300.0, 400.0])

        result = _interpolate_z_in_geometry(polygon, x, y)

        np.testing.assert_allclose(result, expected_z, rtol=1e-10)

    def test_points_outside_polygon(self):
        """Test that points outside the polygon return NaN."""
        coords = np.array(
            [
                [0.0, 0.0, 10.0],
                [1.0, 0.0, 20.0],
                [1.0, 1.0, 30.0],
                [0.0, 1.0, 40.0],
                [0.0, 0.0, 10.0],
            ]
        )
        polygon = Polygon(coords[:, :2])
        polygon = polygon.__class__(coords)

        # Points clearly outside the unit square
        x = np.array([2.0, -1.0, 0.5])
        y = np.array([0.5, 0.5, 2.0])

        result = _interpolate_z_in_geometry(polygon, x, y)

        # All points outside should return NaN
        assert np.all(np.isnan(result))

    def test_mixed_inside_outside_points(self):
        """Test interpolation with mix of inside and outside points."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 2.0, 10.0],
                [0.0, 0.0, 0.0],
            ]
        )
        polygon = Polygon(coords[:, :2])
        polygon = polygon.__class__(coords)

        # Mix of inside and outside points
        x = np.array([1.0, 3.0, 0.5, -1.0])  # inside, outside, inside, outside
        y = np.array([0.5, 0.5, 0.25, 0.5])

        result = _interpolate_z_in_geometry(polygon, x, y)

        # First and third points should have valid values,
        # second and fourth should be NaN
        assert not np.isnan(result[0])  # Inside
        assert np.isnan(result[1])  # Outside
        assert not np.isnan(result[2])  # Inside
        assert np.isnan(result[3])  # Outside

    def test_linear_gradient(self):
        """Test interpolation with a perfect linear gradient."""
        # Create rectangle with linear Z gradient from left (0) to right (100)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 100.0],
                [10.0, 1.0, 100.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        polygon = Polygon(coords[:, :2])
        polygon = polygon.__class__(coords)

        # Test points along the gradient
        x = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        y = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # All at middle Y
        expected_z = np.array([0.0, 25.0, 50.0, 75.0, 100.0])

        result = _interpolate_z_in_geometry(polygon, x, y)

        np.testing.assert_allclose(result, expected_z, rtol=1e-10)

    def test_uniform_z_values(self):
        """Test interpolation when all vertices have the same Z value."""
        coords = np.array(
            [
                [0.0, 0.0, 42.0],
                [1.0, 0.0, 42.0],
                [1.0, 1.0, 42.0],
                [0.0, 1.0, 42.0],
                [0.0, 0.0, 42.0],
            ]
        )
        polygon = Polygon(coords[:, :2])
        polygon = polygon.__class__(coords)

        # Any point inside should return the same Z value
        x = np.array([0.5, 0.25, 0.75])
        y = np.array([0.5, 0.25, 0.75])

        result = _interpolate_z_in_geometry(polygon, x, y)

        expected = np.full(3, 42.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_single_point_query(self):
        """Test interpolation with a single query point."""
        coords = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
                [0.5, 1.0, 3.0],
                [0.0, 0.0, 1.0],
            ]
        )
        polygon = Polygon(coords[:, :2])
        polygon = polygon.__class__(coords)

        x = np.array([0.5])
        y = np.array([0.33])

        result = _interpolate_z_in_geometry(polygon, x, y)

        assert len(result) == 1
        assert not np.isnan(result[0])
        assert 1.0 <= result[0] <= 3.0

    def test_large_number_of_points(self):
        """Test interpolation with many query points at once."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 10.0],
                [10.0, 10.0, 20.0],
                [0.0, 10.0, 10.0],
                [0.0, 0.0, 0.0],
            ]
        )
        polygon = Polygon(coords[:, :2])
        polygon = polygon.__class__(coords)

        # Create a grid of 100 points
        x_grid = np.linspace(1, 9, 10)
        y_grid = np.linspace(1, 9, 10)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        x = x_mesh.flatten()
        y = y_mesh.flatten()

        result = _interpolate_z_in_geometry(polygon, x, y)

        assert len(result) == 100
        # All points should be inside and have valid interpolated values
        assert np.all(~np.isnan(result))
        # All values should be within the expected range
        assert np.all((result >= 0.0) & (result <= 20.0))

    def test_mismatched_array_shapes(self):
        """Test error handling for mismatched x and y array shapes."""
        coords = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
                [1.0, 1.0, 3.0],
                [0.0, 0.0, 1.0],
            ]
        )
        polygon = Polygon(coords[:, :2])
        polygon = polygon.__class__(coords)

        x = np.array([0.5, 0.25])  # 2 elements
        y = np.array([0.5])  # 1 element

        with pytest.raises(
            ValueError,
            match=r"all the input array dimensions except for the concatenation axis "
            r"must match exactly,",
        ):
            _interpolate_z_in_geometry(polygon, x, y)

    def test_empty_arrays(self):
        """Test interpolation with empty input arrays."""
        coords = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
                [1.0, 1.0, 3.0],
                [0.0, 0.0, 1.0],
            ]
        )
        polygon = Polygon(coords[:, :2])
        polygon = polygon.__class__(coords)

        x = np.array([])
        y = np.array([])

        result = _interpolate_z_in_geometry(polygon, x, y)

        assert len(result) == 0
        assert result.dtype == np.float64

    def test_nan_z_values(self):
        """Test interpolation when some Z values are NaN."""
        coords = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, np.nan],  # NaN Z value
                [1.0, 1.0, 3.0],
                [0.0, 1.0, 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        polygon = Polygon(coords[:, :2])
        polygon = polygon.__class__(coords)

        x = np.array([0.5])
        y = np.array([0.5])

        result = _interpolate_z_in_geometry(polygon, x, y)

        # Result should be NaN because one of the boundary values is NaN
        assert np.isnan(result[0])

    def test_complex_polygon_shape(self):
        """Test interpolation with a more complex polygon shape."""
        # Create an L-shaped polygon
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 2.0],
                [2.0, 1.0, 3.0],
                [1.0, 1.0, 2.0],
                [1.0, 2.0, 2.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        polygon = Polygon(coords[:, :2])
        polygon = polygon.__class__(coords)

        # Point inside the L-shape
        x = np.array([0.5])
        y = np.array([0.5])

        result = _interpolate_z_in_geometry(polygon, x, y)

        assert len(result) == 1
        assert not np.isnan(result[0])
        # Should be within the range of Z values
        assert 0.0 <= result[0] <= 3.0

    def test_boundary_edge_points(self):
        """Test interpolation for points exactly on polygon edges."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 20.0],
                [2.0, 2.0, 40.0],
                [0.0, 2.0, 20.0],
                [0.0, 0.0, 0.0],
            ]
        )
        polygon = Polygon(coords[:, :2])
        polygon = polygon.__class__(coords)

        # Points on the bottom edge should interpolate between 0 and 20
        x = np.array([1.0])  # Midpoint of bottom edge
        y = np.array([0.0])

        result = _interpolate_z_in_geometry(polygon, x, y)

        # Should be exactly 10 (midpoint between 0 and 20)
        np.testing.assert_allclose(result, [10.0], rtol=1e-10)

    def test_polygon_with_hole(self):
        """Test that function works with simple polygon (exterior only)."""
        # Note: The function only uses polygon.exterior.coords, so it doesn't handle
        # holes. This test documents that behaviour
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 40.0],
                [4.0, 4.0, 80.0],
                [0.0, 4.0, 40.0],
                [0.0, 0.0, 0.0],
            ]
        )

        # Create polygon with a hole (though function ignores holes)
        exterior = coords[:, :2]
        hole = np.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]])
        polygon = Polygon(exterior, [hole])
        polygon = polygon.__class__(coords)  # Make it 3D

        # Point inside the hole should still get interpolated
        # (because function only considers exterior boundary)
        x = np.array([2.0])
        y = np.array([2.0])

        result = _interpolate_z_in_geometry(polygon, x, y)

        assert not np.isnan(result[0])
        assert 0.0 <= result[0] <= 80.0

    def test_very_small_polygon(self):
        """Test interpolation with a very small polygon."""
        coords = np.array(
            [
                [0.0, 0.0, 1.0],
                [1e-6, 0.0, 2.0],
                [1e-6, 1e-6, 3.0],
                [0.0, 1e-6, 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        polygon = Polygon(coords[:, :2])
        polygon = polygon.__class__(coords)

        # Point inside the tiny polygon
        x = np.array([5e-7])
        y = np.array([5e-7])

        result = _interpolate_z_in_geometry(polygon, x, y)

        assert len(result) == 1
        # Should either be a valid interpolation or NaN (depends on numerical precision)
        if not np.isnan(result[0]):
            assert 1.0 <= result[0] <= 3.0

    def test_output_dtype(self):
        """Test that output array has correct dtype (float64)."""
        coords = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
                [1.0, 1.0, 3.0],
                [0.0, 0.0, 1.0],
            ]
        )
        polygon = Polygon(coords[:, :2])
        polygon = polygon.__class__(coords)

        x = np.array([0.5])
        y = np.array([0.5])

        result = _interpolate_z_in_geometry(polygon, x, y)

        assert result.dtype == np.float64


class TestRasterizeZGDF:
    """Test suite for rasterize_z_gdf function."""

    def _create_3d_polygon(
        self, coords_2d: np.ndarray, z_values: np.ndarray
    ) -> Polygon:
        """Helper to create 3D polygon from 2D coordinates and Z values."""
        coords_3d = np.column_stack([coords_2d, z_values])
        return Polygon(coords_3d)

    def test_basic_z_interpolation_single_polygon(self):
        """Test basic Z interpolation with a single 3D polygon."""
        import geopandas as gpd

        # Create a square with Z values at vertices
        coords_2d = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
                [0.0, 0.0],
            ]
        )
        z_values = np.array([0.0, 10.0, 20.0, 10.0, 0.0])

        polygon = self._create_3d_polygon(coords_2d, z_values)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=_PROJECTED_CRS)

        raster_meta = RasterMeta(
            cell_size=0.5, crs=_PROJECTED_CRS, transform=Affine.scale(0.5, -0.5)
        )
        result = rasterize_z_gdf(
            gdf, cell_size=raster_meta.cell_size, crs=raster_meta.crs
        )
        assert isinstance(result, Raster)
        # The cell size and CRS should be preserved
        assert result.raster_meta.cell_size == raster_meta.cell_size
        assert result.raster_meta.crs == raster_meta.crs

    def test_multiple_polygons_mean_aggregation(self):
        """Test Z interpolation with multiple overlapping polygons using mean
        aggregation."""
        import geopandas as gpd

        # Create two overlapping squares with different Z values
        coords_2d_1 = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
                [0.0, 0.0],
            ]
        )
        z_values_1 = np.array([0.0, 0.0, 10.0, 10.0, 0.0])

        coords_2d_2 = np.array(
            [
                [1.0, 1.0],
                [3.0, 1.0],
                [3.0, 3.0],
                [1.0, 3.0],
                [1.0, 1.0],
            ]
        )
        z_values_2 = np.array([20.0, 20.0, 30.0, 30.0, 20.0])

        polygon1 = self._create_3d_polygon(coords_2d_1, z_values_1)
        polygon2 = self._create_3d_polygon(coords_2d_2, z_values_2)
        gdf = gpd.GeoDataFrame(geometry=[polygon1, polygon2], crs=_PROJECTED_CRS)

        raster_meta = RasterMeta(
            cell_size=0.5, crs=_PROJECTED_CRS, transform=Affine.scale(0.5, -0.5)
        )

        result = rasterize_z_gdf(
            gdf, cell_size=raster_meta.cell_size, crs=raster_meta.crs, agg="mean"
        )

        assert isinstance(result, Raster)
        raster_array = result.arr

        # In overlapping areas, values should be averaged
        # Check that there are valid values
        valid_values = raster_array[~np.isnan(raster_array)]
        assert len(valid_values) > 0
        assert np.all((valid_values >= 0.0) & (valid_values <= 30.0))

    def test_min_aggregation(self):
        """Test Z interpolation with min aggregation for overlapping polygons."""
        import geopandas as gpd

        # Create two overlapping squares with different Z ranges
        coords_2d_1 = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
                [0.0, 0.0],
            ]
        )
        z_values_1 = np.array([10.0, 10.0, 20.0, 20.0, 10.0])

        coords_2d_2 = np.array(
            [
                [1.0, 1.0],
                [3.0, 1.0],
                [3.0, 3.0],
                [1.0, 3.0],
                [1.0, 1.0],
            ]
        )
        z_values_2 = np.array([5.0, 5.0, 15.0, 15.0, 5.0])

        polygon1 = self._create_3d_polygon(coords_2d_1, z_values_1)
        polygon2 = self._create_3d_polygon(coords_2d_2, z_values_2)
        gdf = gpd.GeoDataFrame(geometry=[polygon1, polygon2], crs=_PROJECTED_CRS)

        raster_meta = RasterMeta(
            cell_size=0.5, crs=_PROJECTED_CRS, transform=Affine.scale(0.5, -0.5)
        )

        result = rasterize_z_gdf(
            gdf, cell_size=raster_meta.cell_size, crs=raster_meta.crs, agg="min"
        )

        assert isinstance(result, Raster)
        raster_array = result.arr

        # In overlapping areas, should take minimum values
        valid_values = raster_array[~np.isnan(raster_array)]
        assert len(valid_values) > 0
        assert np.all((valid_values >= 5.0) & (valid_values <= 20.0))

    def test_max_aggregation(self):
        """Test Z interpolation with max aggregation for overlapping polygons."""
        import geopandas as gpd

        # Create two overlapping squares with different Z ranges
        coords_2d_1 = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
                [0.0, 0.0],
            ]
        )
        z_values_1 = np.array([10.0, 10.0, 20.0, 20.0, 10.0])

        coords_2d_2 = np.array(
            [
                [1.0, 1.0],
                [3.0, 1.0],
                [3.0, 3.0],
                [1.0, 3.0],
                [1.0, 1.0],
            ]
        )
        z_values_2 = np.array([5.0, 5.0, 25.0, 25.0, 5.0])

        polygon1 = self._create_3d_polygon(coords_2d_1, z_values_1)
        polygon2 = self._create_3d_polygon(coords_2d_2, z_values_2)
        gdf = gpd.GeoDataFrame(geometry=[polygon1, polygon2], crs=_PROJECTED_CRS)

        raster_meta = RasterMeta(
            cell_size=0.5, crs=_PROJECTED_CRS, transform=Affine.scale(0.5, -0.5)
        )

        result = rasterize_z_gdf(
            gdf, cell_size=raster_meta.cell_size, crs=raster_meta.crs, agg="max"
        )

        assert isinstance(result, Raster)
        raster_array = result.arr

        # In overlapping areas, should take maximum values
        valid_values = raster_array[~np.isnan(raster_array)]
        assert len(valid_values) > 0
        assert np.all((valid_values >= 5.0) & (valid_values <= 25.0))

    def test_empty_geodataframe(self):
        """Test with empty GeoDataFrame."""
        import geopandas as gpd

        gdf = gpd.GeoDataFrame(geometry=[], crs=_PROJECTED_CRS)
        raster_meta = RasterMeta(
            cell_size=1.0, crs=_PROJECTED_CRS, transform=Affine.scale(1.0, -1.0)
        )

        with pytest.raises(
            ValueError, match=r"Cannot rasterize an empty GeoDataFrame."
        ):
            rasterize_z_gdf(gdf, cell_size=raster_meta.cell_size, crs=raster_meta.crs)

    def test_2d_polygons_converted_to_3d(self):
        """Test that 2D polygons are converted to 3D with NaN Z values."""
        import geopandas as gpd

        # Create a 2D polygon (no Z coordinates)
        polygon_2d = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        gdf = gpd.GeoDataFrame(geometry=[polygon_2d], crs=_PROJECTED_CRS)

        raster_meta = RasterMeta(
            cell_size=0.5, crs=_PROJECTED_CRS, transform=Affine.scale(0.5, -0.5)
        )

        # Should raise an error because 2D polygons don't have Z coordinates
        with pytest.raises(ValueError, match="not 3D"):
            rasterize_z_gdf(gdf, cell_size=raster_meta.cell_size, crs=raster_meta.crs)

    def test_mixed_2d_3d_polygons(self):
        """Test with mix of 2D and 3D polygons."""
        import geopandas as gpd

        # Create a 3D polygon
        coords_3d = np.array(
            [
                [0.0, 0.0, 10.0],
                [1.0, 0.0, 20.0],
                [1.0, 1.0, 30.0],
                [0.0, 1.0, 20.0],
                [0.0, 0.0, 10.0],
            ]
        )
        polygon_3d = Polygon(coords_3d)

        # Create a 2D polygon
        polygon_2d = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])

        gdf = gpd.GeoDataFrame(geometry=[polygon_3d, polygon_2d], crs=_PROJECTED_CRS)
        raster_meta = RasterMeta(
            cell_size=0.5, crs=_PROJECTED_CRS, transform=Affine.scale(0.5, -0.5)
        )

        # Should raise an error because not all polygons are 3D
        with pytest.raises(ValueError, match="not 3D"):
            rasterize_z_gdf(gdf, cell_size=raster_meta.cell_size, crs=raster_meta.crs)

    def test_polygons_with_nan_z_values(self):
        """Test polygons where some vertices have NaN Z values."""
        import geopandas as gpd

        coords_2d = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
                [0.0, 0.0],
            ]
        )
        z_values = np.array([10.0, np.nan, 20.0, 15.0, 10.0])  # One NaN Z value

        polygon = self._create_3d_polygon(coords_2d, z_values)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=_PROJECTED_CRS)

        raster_meta = RasterMeta(
            cell_size=0.5, crs=_PROJECTED_CRS, transform=Affine.scale(0.5, -0.5)
        )

        result = rasterize_z_gdf(
            gdf, cell_size=raster_meta.cell_size, crs=raster_meta.crs
        )

        assert isinstance(result, Raster)
        # Due to NaN in boundary, interpolated values should be NaN
        raster_array = result.arr
        # Should have some NaN values due to interpolation issues
        assert np.any(np.isnan(raster_array))

    def test_linear_z_gradient(self):
        """Test with a polygon having a perfect linear Z gradient."""
        import geopandas as gpd

        # Create rectangle with linear gradient from bottom (Z=0) to top (Z=10)
        coords_2d = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
                [0.0, 0.0],
            ]
        )
        z_values = np.array([0.0, 0.0, 10.0, 10.0, 0.0])

        polygon = self._create_3d_polygon(coords_2d, z_values)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=_PROJECTED_CRS)

        raster_meta = RasterMeta(
            cell_size=0.5, crs=_PROJECTED_CRS, transform=Affine.scale(0.5, -0.5)
        )

        result = rasterize_z_gdf(
            gdf, cell_size=raster_meta.cell_size, crs=raster_meta.crs
        )

        assert isinstance(result, Raster)
        raster_array = result.arr
        valid_values = raster_array[~np.isnan(raster_array)]

        # Should have linear gradient values between 0 and 10
        assert len(valid_values) > 0
        assert np.all((valid_values >= 0.0) & (valid_values <= 10.0))
        # Check that there's actually a gradient (not all same value)
        assert np.std(valid_values) > 1.0

    def test_triangular_polygon(self):
        """Test Z interpolation with triangular polygon."""
        import geopandas as gpd

        coords_2d = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [1.0, 2.0],
                [0.0, 0.0],
            ]
        )
        z_values = np.array([0.0, 10.0, 20.0, 0.0])

        polygon = self._create_3d_polygon(coords_2d, z_values)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=_PROJECTED_CRS)

        raster_meta = RasterMeta(
            cell_size=0.25, crs=_PROJECTED_CRS, transform=Affine.scale(0.25, -0.25)
        )

        result = rasterize_z_gdf(
            gdf, cell_size=raster_meta.cell_size, crs=raster_meta.crs
        )

        assert isinstance(result, Raster)
        raster_array = result.arr
        valid_values = raster_array[~np.isnan(raster_array)]

        # Triangle should have interpolated values between min and max
        assert len(valid_values) > 0
        assert np.all((valid_values >= 0.0) & (valid_values <= 20.0))

    def test_large_number_of_polygons(self):
        """Test performance with many polygons."""
        import geopandas as gpd

        polygons = []
        for i in range(50):  # Create 50 small polygons
            x_offset = (i % 10) * 0.5
            y_offset = (i // 10) * 0.5
            coords_2d = np.array(
                [
                    [x_offset, y_offset],
                    [x_offset + 0.4, y_offset],
                    [x_offset + 0.4, y_offset + 0.4],
                    [x_offset, y_offset + 0.4],
                    [x_offset, y_offset],
                ]
            )
            z_values = np.array([i, i + 1, i + 2, i + 1, i])
            polygon = self._create_3d_polygon(coords_2d, z_values)
            polygons.append(polygon)

        gdf = gpd.GeoDataFrame(geometry=polygons, crs=_PROJECTED_CRS)
        raster_meta = RasterMeta(
            cell_size=0.1, crs=_PROJECTED_CRS, transform=Affine.scale(0.1, -0.1)
        )

        result = rasterize_z_gdf(
            gdf, cell_size=raster_meta.cell_size, crs=raster_meta.crs
        )

        assert isinstance(result, Raster)
        raster_array = result.arr
        valid_values = raster_array[~np.isnan(raster_array)]

        # Should have many valid interpolated values
        assert len(valid_values) > 100

    def test_very_small_cell_size(self):
        """Test with very small cell size for high resolution."""
        import geopandas as gpd

        coords_2d = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        z_values = np.array([0.0, 10.0, 20.0, 10.0, 0.0])

        polygon = self._create_3d_polygon(coords_2d, z_values)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=_PROJECTED_CRS)

        raster_meta = RasterMeta(
            cell_size=0.01, crs=_PROJECTED_CRS, transform=Affine.scale(0.01, -0.01)
        )

        result = rasterize_z_gdf(
            gdf, cell_size=raster_meta.cell_size, crs=raster_meta.crs
        )

        assert isinstance(result, Raster)
        # Should create high-resolution raster (100x100 for 1x1 unit)
        assert result.arr.shape[0] >= 100
        assert result.arr.shape[1] >= 100

    def test_output_raster_metadata(self):
        """Test that output raster has correct metadata."""
        import geopandas as gpd

        coords_2d = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        z_values = np.array([0.0, 10.0, 20.0, 10.0, 0.0])

        polygon = self._create_3d_polygon(coords_2d, z_values)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=_PROJECTED_CRS)

        custom_transform = Affine.scale(0.5, -0.5) * Affine.translation(100, 200)
        raster_meta = RasterMeta(
            cell_size=0.5, crs=_PROJECTED_CRS, transform=custom_transform
        )

        result = rasterize_z_gdf(
            gdf, cell_size=raster_meta.cell_size, crs=raster_meta.crs
        )

        assert result.raster_meta.cell_size == 0.5
        assert result.raster_meta.crs == _PROJECTED_CRS
        # Note: The output transform may be modified due to bounds expansion

    def test_output_dtype_float64(self):
        """Test that output raster has float64 dtype."""
        import geopandas as gpd

        coords_2d = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        z_values = np.array([0.0, 10.0, 20.0, 10.0, 0.0])

        polygon = self._create_3d_polygon(coords_2d, z_values)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=_PROJECTED_CRS)

        raster_meta = RasterMeta(
            cell_size=0.5, crs=_PROJECTED_CRS, transform=Affine.scale(0.5, -0.5)
        )

        result = rasterize_z_gdf(
            gdf, cell_size=raster_meta.cell_size, crs=raster_meta.crs
        )

        assert result.arr.dtype == np.float64

    def test_non_overlapping_polygons(self):
        """Test with non-overlapping polygons (no aggregation needed)."""
        import geopandas as gpd

        # Create two separate squares
        coords_2d_1 = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        z_values_1 = np.array([0.0, 5.0, 10.0, 5.0, 0.0])

        coords_2d_2 = np.array(
            [
                [2.0, 0.0],
                [3.0, 0.0],
                [3.0, 1.0],
                [2.0, 1.0],
                [2.0, 0.0],
            ]
        )
        z_values_2 = np.array([20.0, 25.0, 30.0, 25.0, 20.0])

        polygon1 = self._create_3d_polygon(coords_2d_1, z_values_1)
        polygon2 = self._create_3d_polygon(coords_2d_2, z_values_2)
        gdf = gpd.GeoDataFrame(geometry=[polygon1, polygon2], crs=_PROJECTED_CRS)

        raster_meta = RasterMeta(
            cell_size=0.25, crs=_PROJECTED_CRS, transform=Affine.scale(0.25, -0.25)
        )

        result = rasterize_z_gdf(
            gdf, cell_size=raster_meta.cell_size, crs=raster_meta.crs
        )

        assert isinstance(result, Raster)
        raster_array = result.arr
        valid_values = raster_array[~np.isnan(raster_array)]

        # Should have values from both polygons
        assert len(valid_values) > 0
        # Values should span the range of both polygons
        assert np.min(valid_values) <= 10.0  # From first polygon
        assert np.max(valid_values) >= 20.0  # From second polygon

    def test_gaps_between_polygons(self):
        """Test that gaps between polygons result in NaN values."""
        import geopandas as gpd

        # Create two separate squares with a gap between them
        coords_2d_1 = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        z_values_1 = np.array([0.0, 5.0, 10.0, 5.0, 0.0])

        coords_2d_2 = np.array(
            [
                [3.0, 0.0],  # Gap from x=1 to x=3
                [4.0, 0.0],
                [4.0, 1.0],
                [3.0, 1.0],
                [3.0, 0.0],
            ]
        )
        z_values_2 = np.array([20.0, 25.0, 30.0, 25.0, 20.0])

        polygon1 = self._create_3d_polygon(coords_2d_1, z_values_1)
        polygon2 = self._create_3d_polygon(coords_2d_2, z_values_2)
        gdf = gpd.GeoDataFrame(geometry=[polygon1, polygon2], crs=_PROJECTED_CRS)

        raster_meta = RasterMeta(
            cell_size=0.25, crs=_PROJECTED_CRS, transform=Affine.scale(0.25, -0.25)
        )

        result = rasterize_z_gdf(
            gdf, cell_size=raster_meta.cell_size, crs=raster_meta.crs
        )

        assert isinstance(result, Raster)
        raster_array = result.arr

        # There should be NaN values in the gap area
        assert np.any(np.isnan(raster_array))

        # But also valid values where polygons exist
        valid_values = raster_array[~np.isnan(raster_array)]
        assert len(valid_values) > 0

    def test_bounds_preservation_no_resampling(self):
        """Bounds of raster should match input GeoDataFrame shifted by half a cell."""
        import geopandas as gpd

        coords_2d = np.array(
            [
                [100.0, 200.0],
                [150.0, 200.0],
                [150.0, 250.0],
                [100.0, 250.0],
                [100.0, 200.0],
            ]
        )
        z_values = np.array([0.0, 10.0, 20.0, 10.0, 0.0])

        polygon = self._create_3d_polygon(coords_2d, z_values)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=_PROJECTED_CRS)

        # Create raster with specific bounds
        cell_size = 5.0

        result = rasterize_z_gdf(gdf, cell_size=cell_size, crs=_PROJECTED_CRS)

        # Get the bounds of the result
        result_bounds = result.bounds

        # With the new infer implementation, bounds extend by cell_size/2 on each side
        # to ensure cell centers align with data points
        gdf_bounds = gdf.total_bounds
        expected_bounds = (
            gdf_bounds[0] - cell_size / 2,  # minx - half cell
            gdf_bounds[1] - cell_size / 2,  # miny - half cell
            gdf_bounds[2] + cell_size / 2,  # maxx + half cell
            gdf_bounds[3] + cell_size / 2,  # maxy + half cell
        )

        # Check bounds are within relative tolerance
        rel_tolerance = 1e-4

        np.testing.assert_allclose(
            result_bounds,
            expected_bounds,
            rtol=rel_tolerance,
            err_msg="Bounds should be preserved within relative tolerance",
        )

    def test_simple_box_exact_array(self):
        """Test rasterize_z_gdf with exact array comparison for a simple box."""
        import geopandas as gpd

        # Create a simple 1x1 box with Z-values at corners
        coords_2d = np.array(
            [
                [0.0, 0.0],  # Bottom-left: Z=0
                [1.0, 0.0],  # Bottom-right: Z=1
                [1.0, 1.0],  # Top-right: Z=2
                [0.0, 1.0],  # Top-left: Z=1
                [0.0, 0.0],  # Close polygon
            ]
        )
        z_values = np.array([0.0, 1.0, 2.0, 1.0, 0.0])

        polygon = self._create_3d_polygon(coords_2d, z_values)
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs=_PROJECTED_CRS)

        # Use cell size 0.5 for the unit square
        cell_size = 0.5

        result = rasterize_z_gdf(gdf, cell_size=cell_size, crs=_PROJECTED_CRS)

        # Shape is (3, 3) with cell centers at:
        # - Columns (x): 0.0, 0.5, 1.0
        # - Rows (y): 1.0, 0.5, 0.0
        # This grid captures all corner points exactly plus interpolated midpoints

        expected_array = np.array(
            [
                [1.0, 1.5, 2.0],
                [0.5, 1.0, 1.5],
                [0.0, 0.5, 1.0],
            ],
            dtype=np.float64,
        )

        # Test against expected array using allclose
        np.testing.assert_allclose(
            result.arr,
            expected_array,
            rtol=1e-5,
            err_msg="Interpolated array doesn't match expected values",
            strict=True,
        )


class TestRasterFromPointCloud:
    def test_square(self):
        """Test rasterization from a simple square point cloud."""

        # Arrange
        x = [0, 0, 1, 1]
        y = [0, 1, 0, 1]
        z = [10, 20, 30, 40]

        # Act
        raster = raster_from_point_cloud(x=x, y=y, z=z, crs="EPSG:2193", cell_size=0.5)

        # Assert
        assert isinstance(raster, Raster)
        assert raster.arr.shape == (3, 3)

        # The non-corner cells are interpolated
        expected_array = np.array(
            [
                [
                    20.0,
                    (20 + 40) / 2,
                    40.0,
                ],
                [
                    (10 + 20) / 2,
                    (10 + 20 + 30 + 40) / 4,
                    (40 + 30) / 2,
                ],
                [10.0, (10 + 30) / 2, 30.0],
            ]
        )
        np.testing.assert_array_equal(raster.arr, expected_array)

    def test_cell_size_heuristic(self):
        """Test that the cell size heuristic works as expected."""
        # Arrange
        x = [0, 0, 1, 1]
        y = [0, 1, 0, 1]
        z = [10, 20, 30, 40]

        # Act
        raster = raster_from_point_cloud(x=x, y=y, z=z, crs="EPSG:2193")

        # Assert
        assert isinstance(raster, Raster)
        assert raster.arr.shape == (3, 3)

    class TestLengthMismatch:
        def test_xy(self):
            # Arrange
            x = [0, 0, 1]
            y = [0, 1, 0, 1]
            z = [10, 20, 30]

            # Act / Assert
            with pytest.raises(
                ValueError, match=r"Length of x, y, and z must be equal\."
            ):
                raster_from_point_cloud(x=x, y=y, z=z, crs="EPSG:2193")

        def test_xz(self):
            # Arrange
            x = [0, 0, 1]
            y = [0, 1, 0]
            z = [10, 20, 30, 40]

            # Act / Assert
            with pytest.raises(
                ValueError, match=r"Length of x, y, and z must be equal\."
            ):
                raster_from_point_cloud(x=x, y=y, z=z, crs="EPSG:2193")

    def test_random_point_cloud(self):
        # Arrange
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 100, size=1000)
        y = rng.uniform(0, 100, size=1000)
        z = rng.uniform(0, 1000, size=1000)

        # Act
        raster = raster_from_point_cloud(
            x=x, y=y, z=z, crs="EPSG:2193", cell_size=5.0
        ).extrapolate()  # Fill NaNs at the edges

        # Assert
        assert isinstance(raster, Raster)
        assert raster.arr.shape == (21, 21)
        assert np.all(raster.arr >= 0)  # All values should be non-negative
        assert np.all(raster.arr <= 1000)  # All values should be within z range
        assert 450 < raster.arr.mean() < 550  # Mean should be roughly accurate

    def test_same_xy_different_z(self):
        # Arrange
        x = [0, 0, 0, 1, 1]
        y = [0, 0, 1, 0, 1]
        z = [10, 15, 20, 30, 40]  # Two points at (0,0) with different z

        # Act
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Duplicate (x, y) points found. Each (x, y) point must be unique."
            ),
        ):
            raster_from_point_cloud(x=x, y=y, z=z, crs="EPSG:2193")

    def test_same_xyz_triples(self):
        # Arrange - duplicate (x,y,z) triples should be accepted and deduplicated
        x = [0, 0, 0, 1, 1]
        y = [0, 0, 1, 0, 1]
        z = [10, 10, 20, 30, 40]  # Two identical points at (0,0,10)

        # Act
        raster = raster_from_point_cloud(x=x, y=y, z=z, crs="EPSG:2193", cell_size=0.5)

        # Assert
        assert isinstance(raster, Raster)
        # Should successfully create a raster with deduplicated points
        assert raster.arr.shape == (3, 3)

    def test_multiple_xyz_duplicates(self):
        # Arrange - multiple duplicate (x,y,z) triples should all be deduplicated
        x = [0, 0, 0, 1, 1, 1, 1]
        y = [0, 1, 1, 0, 1, 1, 1]
        z = [10, 20, 20, 30, 40, 40, 40]  # (0,1,20) duplicated, (1,1,40) tripled

        # Act
        raster = raster_from_point_cloud(x=x, y=y, z=z, crs="EPSG:2193", cell_size=0.5)

        # Assert
        assert isinstance(raster, Raster)
        assert raster.arr.shape == (3, 3)

    def test_collinear_points(self):
        # Arrange
        x = [0, 1, 2]
        y = [0, 0, 0]  # All points are collinear along y=0
        z = [10, 20, 30]

        # Act / Assert
        with pytest.raises(
            ValueError,
            match=re.escape("Failed to interpolate."),
        ):
            raster_from_point_cloud(x=x, y=y, z=z, crs="EPSG:2193")

    class TestInsufficientPoints:
        def test_empty_inputs(self):
            # Arrange
            x: list[float] = []
            y: list[float] = []
            z: list[float] = []

            # Act / Assert
            with pytest.raises(
                ValueError,
                match=re.escape("At least three valid (x, y, z) points are required"),
            ):
                raster_from_point_cloud(x=x, y=y, z=z, crs="EPSG:2193")

        def test_only_two_points(self):
            # Arrange
            x = [0, 1]
            y = [0, 1]
            z = [10, 20]

            # Act / Assert
            with pytest.raises(
                ValueError,
                match=re.escape("At least three valid (x, y, z) points are required"),
            ):
                raster_from_point_cloud(x=x, y=y, z=z, crs="EPSG:2193")

    def test_2d_arrays(self):
        # Arrange
        x = np.array([[0, 0], [1, 1]])
        y = np.array([[0, 1], [0, 1]])
        z = np.array([[10, 20], [30, 40]])

        # Act
        raster = raster_from_point_cloud(x=x, y=y, z=z, crs="EPSG:2193", cell_size=0.5)

        # Assert
        assert isinstance(raster, Raster)
        assert raster.arr.shape == (3, 3)

    def test_xy_are_nan_warns(self):
        # Arrange
        x = [0, 0, np.nan, 1, 3]
        y = [0, 1, 0, np.nan, 2]
        z = [10, 20, 30, 40, 50]

        # Act
        with pytest.warns(
            UserWarning,
            match=re.escape(
                "Some (x,y) points are NaN-valued or non-finite. These will be ignored."
            ),
        ):
            raster = raster_from_point_cloud(x=x, y=y, z=z, crs="EPSG:2193")

        # Assert
        assert isinstance(raster, Raster)

    def test_xy_are_infinite_warns(self):
        # Arrange
        x = [0, 0, np.inf, 1, 3]
        y = [0, 1, 0, -np.inf, 2]
        z = [10, 20, 30, 40, 50]

        # Act
        with pytest.warns(
            UserWarning,
            match=re.escape(
                "Some (x,y) points are NaN-valued or non-finite. These will be ignored."
            ),
        ):
            raster = raster_from_point_cloud(x=x, y=y, z=z, crs="EPSG:2193")

        # Assert
        assert isinstance(raster, Raster)

    def test_z_is_nan(self):
        # This works fine, it just means any concave
        # area with NaN z values will be NaN in the output

        # Arrange
        x = [0, 0, 1, 1]
        y = [0, 1, 0, 1]
        z = [10, np.nan, 30, 40]

        # Act
        raster = raster_from_point_cloud(x=x, y=y, z=z, crs="EPSG:2193", cell_size=0.5)

        # Assert
        assert isinstance(raster, Raster)

    def test_less_than_three_valid_points_due_to_nan(self):
        # We want a good error message if there are

        # Arrange
        x = [0, np.nan, 1]
        y = [0, 1, np.nan]
        z = [10, 20, 30]

        # Act / Assert
        with (
            pytest.raises(
                ValueError,
                match=re.escape("At least three valid (x, y, z) points are required"),
            ),
            pytest.warns(
                UserWarning,
                match=re.escape(
                    "Some (x,y) points are NaN-valued or non-finite. "
                    "These will be ignored."
                ),
            ),
        ):
            raster_from_point_cloud(x=x, y=y, z=z, crs="EPSG:2193")


class TestRasterFromContours:
    def test_linear_contours(self):
        line_bottom = LineString([(0.0, 0.0), (3.0, 0.0)])
        line_top = LineString([(0.0, 3.0), (3.0, 3.0)])

        values = [0.0, 30.0]
        geometry = [line_bottom, line_top]

        # expected values based on linear interpolation between 0 (bottom) and 30 (top)
        expected = np.array(
            [
                [30.0, 30.0, 30.0, 30.0],
                [20.0, 20.0, 20.0, 20.0],
                [10.0, 10.0, 10.0, 10.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        result = raster_from_contours(
            values=values, geometry=geometry, crs=_PROJECTED_CRS, cell_size=1.0
        )

        # Check metadata
        assert isinstance(result, Raster)
        assert result.cell_size == 1.0
        assert result.crs == _PROJECTED_CRS

        # Check that raster is as expected
        raster_array = result.arr
        np.testing.assert_array_almost_equal(raster_array, expected)

    def test_mismatched_lengths(self):
        line1 = LineString([(0.0, 0.0), (1.0, 1.0)])
        line2 = LineString([(1.0, 0.0), (2.0, 1.0)])

        values = [10.0, 20.0, 30.0]  # 3 values
        geometry = [line1, line2]  # 2 geometries

        with pytest.raises(
            ValueError, match="Values and geometry must have the same length"
        ):
            raster_from_contours(
                values=values,
                geometry=geometry,
                crs=_PROJECTED_CRS,
                cell_size=1.0,
            )

    def test_length_zero_arrays(self):
        values = []
        geometry = []

        with pytest.raises(
            ValueError, match="At least two distinct contour values are required"
        ):
            raster_from_contours(
                values=values,
                geometry=geometry,
                crs=_PROJECTED_CRS,
                cell_size=1.0,
            )

    def test_single_value_only(self):
        line = LineString([(0.0, 0.0), (1.0, 1.0)])
        values = [10.0]  # Only 1 value
        geometry = [line]

        with pytest.raises(
            ValueError, match="At least two distinct contour values are required"
        ):
            raster_from_contours(
                values=values,
                geometry=geometry,
                crs=_PROJECTED_CRS,
                cell_size=1.0,
            )

    def test_three_identical_values(self):
        line1 = LineString([(0.0, 0.0), (1.0, 1.0)])
        line2 = LineString([(1.0, 0.0), (2.0, 1.0)])
        line3 = LineString([(2.0, 0.0), (3.0, 1.0)])

        values = [5.0, 5.0, 5.0]  # All same value
        geometry = [line1, line2, line3]

        with pytest.raises(
            ValueError, match="At least two distinct contour values are required"
        ):
            raster_from_contours(
                values=values,
                geometry=geometry,
                crs=_PROJECTED_CRS,
                cell_size=1.0,
            )

    def test_duplicate_values(self):
        # Two separate circles with the same value
        line1 = LineString([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)])
        line2 = LineString([(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0), (2.0, 0.0)])
        line3 = LineString([(4.0, 0.0), (5.0, 0.0), (5.0, 1.0), (4.0, 1.0), (4.0, 0.0)])

        values = [15.0, 15.0, 20.0]  # Two contours with same value
        geometry = [line1, line2, line3]

        # Should not raise an error despite having duplicate values in geometry
        result = raster_from_contours(
            values=values,
            geometry=geometry,
            crs=_PROJECTED_CRS,
            cell_size=0.5,
        )

        assert isinstance(result, Raster)
        raster_array = result.arr
        valid_values = raster_array[~np.isnan(raster_array)]

        # All valid values should be between 15.0 and 20.0
        assert len(valid_values) > 0
        assert np.min(valid_values) >= 15.0
        assert np.max(valid_values) <= 20.0

    def test_self_intersecting(self):
        # This is a bowtie shape that intersects itself
        self_intersecting = LineString(
            [(0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0), (0.0, 0.0)]
        )
        line2 = LineString([(3.0, 0.0), (4.0, 1.0), (3.0, 2.0)])

        values = [25.0, 35.0]
        geometry = [self_intersecting, line2]

        # Should not raise an error
        result = raster_from_contours(
            values=values,
            geometry=geometry,
            crs=_PROJECTED_CRS,
            cell_size=0.5,
        )

        assert isinstance(result, Raster)
        raster_array = result.arr
        valid_values = raster_array[~np.isnan(raster_array)]

        # Values should fall in expected range
        assert len(valid_values) > 0
        assert np.min(valid_values) >= 25.0
        assert np.max(valid_values) <= 35.0

    def test_polygonal_contours(self):
        # Use Polygon geometries instead of LineString
        polygon1 = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
        polygon2 = Polygon([(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)])

        values = [12.0, 24.0]
        geometry = [polygon1, polygon2]

        # Should not raise an error with Polygon geometries
        result = raster_from_contours(
            values=values,
            geometry=geometry,
            crs=_PROJECTED_CRS,
            cell_size=0.5,
        )

        assert isinstance(result, Raster)
        raster_array = result.arr
        valid_values = raster_array[~np.isnan(raster_array)]

        # Values should fall in expected range
        assert len(valid_values) > 0
        assert np.min(valid_values) >= 12.0
        assert np.max(valid_values) <= 24.0

    def test_multilinestring(self):
        # Create MultiLineString with two separate line segments
        line1a = LineString([(0.0, 0.0), (1.0, 0.0)])
        line1b = LineString([(0.0, 1.0), (1.0, 1.0)])
        multi_line1 = MultiLineString([line1a, line1b])

        line2a = LineString([(2.0, 0.0), (3.0, 0.0)])
        line2b = LineString([(2.0, 1.0), (3.0, 1.0)])
        multi_line2 = MultiLineString([line2a, line2b])

        values = [18.0, 32.0]
        geometry = [multi_line1, multi_line2]

        # Should not raise an error with MultiLineString geometries
        result = raster_from_contours(
            values=values,
            geometry=geometry,
            crs=_PROJECTED_CRS,
            cell_size=0.5,
        )

        assert isinstance(result, Raster)
        raster_array = result.arr
        valid_values = raster_array[~np.isnan(raster_array)]

        # Values should fall in expected range
        assert len(valid_values) > 0
        assert np.min(valid_values) >= 18.0
        assert np.max(valid_values) <= 32.0

    def test_linearring_contours(self):
        # Create LinearRing geometries
        ring1 = LinearRing([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
        ring2 = LinearRing([(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)])

        values = [14.0, 28.0]
        geometry = [ring1, ring2]

        # Should not raise an error with LinearRing geometries
        result = raster_from_contours(
            values=values,
            geometry=geometry,
            crs=_PROJECTED_CRS,
            cell_size=0.5,
        )

        assert isinstance(result, Raster)
        raster_array = result.arr
        valid_values = raster_array[~np.isnan(raster_array)]

        # Values should fall in expected range
        assert len(valid_values) > 0
        assert np.min(valid_values) >= 14.0
        assert np.max(valid_values) <= 28.0

    def test_point_contours(self):
        # Create Point geometries
        point1 = Point(1.0, 1.0)
        point2 = Point(3.0, 5.0)
        point3 = Point(2.0, -2.0)

        values = [50.0, 100.0, 75.0]
        geometry = [point1, point2, point3]

        # Should not raise an error with Point geometries
        result = raster_from_contours(
            values=values,
            geometry=geometry,
            crs=_PROJECTED_CRS,
            cell_size=0.5,
        )

        assert isinstance(result, Raster)
        raster_array = result.arr
        valid_values = raster_array[~np.isnan(raster_array)]

        # Values should fall in expected range
        assert len(valid_values) > 0
        assert np.min(valid_values) >= 50.0
        assert np.max(valid_values) <= 100.0

    def test_kissing_contours(self):
        """Test two contours that touch at a single point."""
        # Arrange
        line1 = LineString([(0, 0), (2, 1)])
        line2 = LineString([(2, 1), (4, 3)])

        # Act
        result = raster_from_contours(
            values=[10.0, 20.0],
            geometry=[line1, line2],
            crs=_PROJECTED_CRS,
            cell_size=0.5,
        )

        # Assert
        assert isinstance(result, Raster)
        # At the touching point (2, 1), the value should be the average: (10 + 20) / 2
        x_coords, y_coords = result.get_xy()
        point_mask = np.isclose(x_coords, 2.0) & np.isclose(y_coords, 1.0)
        point_value = result.arr[point_mask]
        assert point_value.size > 0
        assert np.isclose(point_value, 15.0)

    def test_float_speckling(self, assets_dir: Path):
        """When interpolating between contours, we can get whole areas which all get
        their cell value imputed from the same contour line when we're in a 'pocket',
        e.g. in a horse-shoe shaped contour. However, since we often want to compare
        our raster with the original contours, this can cause a problem where these
        cells which are _meant_ to be the same, actually vary slightly due to floating
        point precision issues, and so when compared against the contour threshold
        lead to speckling effects/noise for visualization.

        This test ensures that we apply a floating point rounding step to the contours
        at a level of precision determined by the input contour values, to ensure that
        such speckling does not occur.
        """
        import geopandas as gpd

        # Arrange
        contour_gdf = gpd.read_parquet(assets_dir / "contour_speckle.parquet")

        # Act
        result = raster_from_contours(
            values=contour_gdf["Contour"],
            geometry=contour_gdf.geometry,
            cell_size=1.0,
        )

        # Assert
        # All values in the horseshoe pocket (above a y-threshold) should be identical
        y_thresh = 5918541.61
        _x, y = result.get_xy()
        pocket_mask = y > y_thresh

        pocket_values = result.arr[pocket_mask]
        unique_values = np.unique(pocket_values[~np.isnan(pocket_values)])
        assert len(unique_values) == 1

    def test_segmentization(self):
        """If we don't segmentize long contour lines, i.e. add extra vertices along
        them, we can get artifacts in the rasterization where long straight lines
        aren't treated as continuous lines and just far-spaced points.
        """

        long_contour = LineString([(0, 0), (0, 100)])
        protected_point = Point(1, 50)
        exposed_contour = LineString([(-5, 45), (-5, 55)])

        values = [10.0, 20.0, 10.0]
        geometry = [long_contour, protected_point, exposed_contour]

        result = raster_from_contours(
            values=values,
            geometry=geometry,
            crs=_PROJECTED_CRS,
            cell_size=1,
        )

        assert isinstance(result, Raster)
        raster_array = result.arr
        # With segmentization, we would expect everything below x=-1 to have
        # value 10.0
        x_thresh = -1
        x, _y = result.get_xy()
        left_mask = x < x_thresh
        left_values = raster_array[left_mask & ~np.isnan(raster_array)]
        assert np.all(left_values == 10.0)


class TestExtractCoords:
    def test_linestring_2d(self):
        line = LineString([(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)])
        coords = list(_extract_coords(line))

        expected = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        assert coords == expected

    def test_polygon_2d(self):
        polygon = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        coords = list(_extract_coords(polygon))

        expected = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        assert coords == expected

    def test_point_2d(self):
        point = Point(1.0, 2.0)
        coords = list(_extract_coords(point))

        expected = [(1.0, 2.0)]
        assert coords == expected

    def test_multilinestring_2d(self):
        line1 = LineString([(0.0, 0.0), (1.0, 1.0)])
        line2 = LineString([(2.0, 0.0), (3.0, 1.0)])
        multi_line = MultiLineString([line1, line2])

        coords = list(_extract_coords(multi_line))

        expected = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]
        assert coords == expected

    def test_multipolygon_2d(self):
        poly1 = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        poly2 = Polygon([(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)])
        multi_poly = MultiPolygon([poly1, poly2])

        coords = list(_extract_coords(multi_poly))

        # Each polygon's exterior includes closing point
        assert len(coords) == 10
        assert coords[0] == (0.0, 0.0)
        assert coords[4] == (0.0, 0.0)  # First polygon closes
        assert coords[5] == (2.0, 0.0)  # Second polygon starts

    def test_multipoint_2d(self):
        point1 = Point(0.0, 0.0)
        point2 = Point(1.0, 1.0)
        point3 = Point(2.0, 0.0)
        multi_point = MultiPoint([point1, point2, point3])

        coords = list(_extract_coords(multi_point))

        expected = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        assert coords == expected

    def test_multipoint_3d(self):
        point1 = Point(0.0, 0.0, 100.0)
        point2 = Point(1.0, 1.0, 200.0)
        point3 = Point(2.0, 0.0, 150.0)
        multi_point = MultiPoint([point1, point2, point3])

        coords = list(_extract_coords(multi_point))

        expected = [(0.0, 0.0, 100.0), (1.0, 1.0, 200.0), (2.0, 0.0, 150.0)]
        assert coords == expected

    def test_unsupported_geometry_type(self):
        geom_collection = GeometryCollection(
            [LineString([(0, 0), (1, 1)]), Point(2, 2)]
        )

        with pytest.raises(NotImplementedError):
            list(_extract_coords(geom_collection))
