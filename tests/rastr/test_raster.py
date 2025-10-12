from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from affine import Affine
from pydantic import ValidationError
from pyproj.crs.crs import CRS
from shapely import MultiPolygon, box
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from rastr.meta import RasterMeta
from rastr.raster import Raster

if TYPE_CHECKING:
    from pathlib import Path

    import folium


@pytest.fixture
def example_raster():
    meta = RasterMeta(
        cell_size=1.0,
        crs=CRS.from_epsg(2193),
        transform=Affine(2.0, 0.0, 0.0, 0.0, 2.0, 0.0),
    )
    arr = np.array([[1, 2], [3, 4]], dtype=float)

    return Raster(arr=arr, raster_meta=meta)


@pytest.fixture
def example_neg_scaled_raster():
    meta = RasterMeta(
        cell_size=1.0,
        crs=CRS.from_epsg(2193),
        transform=Affine(2.0, 0.0, 0.0, 0.0, -2.0, 0.0),
    )
    arr = np.array([[1, 2], [3, 4]])

    return Raster(arr=arr, raster_meta=meta)


@pytest.fixture
def example_raster_with_zeros():
    meta = RasterMeta(
        cell_size=1.0,
        crs=CRS.from_epsg(2193),
        transform=Affine(2.0, 0.0, 0.0, 0.0, 2.0, 0.0),
    )
    arr = np.array([[1, 0], [0, 4]], dtype=float)

    return Raster(
        arr=arr,
        raster_meta=meta,
    )


@pytest.fixture
def stats_test_raster() -> Raster:
    meta = RasterMeta(
        cell_size=1.0,
        crs=CRS.from_epsg(2193),
        transform=Affine(2.0, 0.0, 0.0, 0.0, 2.0, 0.0),
    )
    # Create an array with known statistics
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    return Raster(arr=arr, raster_meta=meta)


@pytest.fixture
def stats_test_raster_with_nans() -> Raster:
    meta = RasterMeta(
        cell_size=1.0,
        crs=CRS.from_epsg(2193),
        transform=Affine(2.0, 0.0, 0.0, 0.0, 2.0, 0.0),
    )
    # Create an array with NaNs to test nan-aware functions
    arr = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
    return Raster(arr=arr, raster_meta=meta)


class TestRaster:
    class TestInit:
        def test_meta_and_arr(self, example_raster: Raster):
            # Act, Assert
            Raster(
                arr=example_raster.arr,
                meta=example_raster.raster_meta,
            )

        def test_both_meta_and_raster_meta(self, example_raster: Raster):
            # Act, Assert
            with pytest.raises(
                ValueError,
                match="Only one of 'meta' or 'raster_meta' should be provided",
            ):
                Raster(
                    arr=example_raster.arr,
                    meta=example_raster.raster_meta,
                    raster_meta=example_raster.raster_meta,
                )

        def test_missing_meta(self, example_raster: Raster):
            # Act, Assert
            with pytest.raises(
                ValueError, match=r"The attribute 'raster_meta' is required\."
            ):
                Raster(arr=example_raster.arr)

    class TestMetaAlias:
        def test_meta_getter(self, example_raster: Raster):
            # Act
            meta_via_alias = example_raster.meta
            meta_direct = example_raster.raster_meta

            # Assert
            assert meta_via_alias is meta_direct
            assert meta_via_alias == meta_direct

        def test_meta_setter(self, example_raster: Raster):
            # Arrange
            example_raster = example_raster.model_copy(deep=True)
            new_meta = RasterMeta(
                cell_size=2.0,
                crs=CRS.from_epsg(4326),
                transform=Affine(1.0, 0.0, 5.0, 0.0, 1.0, 10.0),
            )
            original_meta = example_raster.raster_meta

            # Act
            example_raster.meta = new_meta

            # Assert
            assert example_raster.raster_meta is new_meta
            assert example_raster.meta is new_meta
            assert example_raster.raster_meta != original_meta

    class TestIsLike:
        def test_identical_rasters_are_like(self, example_raster: Raster):
            """Test that a raster is like itself."""
            # Act & Assert
            assert example_raster.is_like(example_raster)

        def test_same_meta_and_shape_are_like(self, example_raster: Raster):
            """Test rasters with same meta and shape but different data are like."""
            # Arrange
            different_arr = np.array([[5, 6], [7, 8]], dtype=float)
            other_raster = Raster(
                arr=different_arr, raster_meta=example_raster.raster_meta
            )

            # Act & Assert
            assert example_raster.is_like(other_raster)
            assert other_raster.is_like(example_raster)

        def test_different_meta_not_like(self, example_raster: Raster):
            """Test that rasters with different meta are not like."""
            # Arrange
            different_meta = RasterMeta(
                cell_size=2.0,
                crs=CRS.from_epsg(4326),
                transform=Affine(1.0, 0.0, 5.0, 0.0, 1.0, 10.0),
            )
            other_raster = Raster(arr=example_raster.arr, raster_meta=different_meta)

            # Act & Assert
            assert not example_raster.is_like(other_raster)
            assert not other_raster.is_like(example_raster)

        def test_different_shape_not_like(self, example_raster: Raster):
            """Test that rasters with different shapes are not like."""
            # Arrange
            # 2x3 instead of 2x2
            different_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
            other_raster = Raster(
                arr=different_arr, raster_meta=example_raster.raster_meta
            )

            # Act & Assert
            assert not example_raster.is_like(other_raster)
            assert not other_raster.is_like(example_raster)

        def test_different_meta_and_shape_not_like(self, example_raster: Raster):
            """Test rasters with both different meta and shape are not like."""
            # Arrange
            different_meta = RasterMeta(
                cell_size=3.0,
                crs=CRS.from_epsg(4326),
                transform=Affine(2.0, 0.0, 10.0, 0.0, 2.0, 20.0),
            )
            different_arr = np.array([[1]], dtype=float)  # 1x1 instead of 2x2
            other_raster = Raster(arr=different_arr, raster_meta=different_meta)

            # Act & Assert
            assert not example_raster.is_like(other_raster)
            assert not other_raster.is_like(example_raster)

    class TestShape:
        def test_shape_property(self, example_raster: Raster):
            # Act
            shape = example_raster.shape

            # Assert
            assert shape == (2, 2)
            assert shape == example_raster.arr.shape

    class TestCRS:
        def test_crs_getter(self, example_raster: Raster):
            # Act
            crs_via_property = example_raster.crs
            crs_via_meta = example_raster.meta.crs
            crs_via_raster_meta = example_raster.raster_meta.crs

            # Assert
            assert crs_via_property is crs_via_meta
            assert crs_via_property is crs_via_raster_meta
            assert crs_via_property == crs_via_meta
            assert crs_via_property == crs_via_raster_meta
            assert isinstance(crs_via_property, CRS)

        def test_crs_setter(self, example_raster: Raster):
            # Arrange
            new_crs = CRS.from_epsg(4326)
            original_crs = example_raster.crs

            # Act
            example_raster.crs = new_crs

            # Assert
            assert example_raster.crs is new_crs
            assert example_raster.meta.crs is new_crs
            assert example_raster.raster_meta.crs is new_crs
            assert example_raster.crs != original_crs

    class TestTransform:
        def test_transform_getter(self, example_raster: Raster):
            # Act
            transform_via_property = example_raster.transform
            transform_via_meta = example_raster.meta.transform
            transform_via_raster_meta = example_raster.raster_meta.transform

            # Assert
            assert transform_via_property is transform_via_meta
            assert transform_via_property is transform_via_raster_meta
            assert transform_via_property == transform_via_meta
            assert transform_via_property == transform_via_raster_meta
            assert isinstance(transform_via_property, Affine)

        def test_transform_setter(self, example_raster: Raster):
            # Arrange
            new_transform = Affine.scale(3.0, 3.0) * Affine.translation(10.0, 20.0)
            original_transform = example_raster.transform

            # Act
            example_raster.transform = new_transform

            # Assert
            assert example_raster.transform is new_transform
            assert example_raster.meta.transform is new_transform
            assert example_raster.raster_meta.transform is new_transform
            assert example_raster.transform != original_transform

    class TestCellSize:
        def test_cell_size_getter(self, example_raster: Raster):
            # Act
            cell_size_via_property = example_raster.cell_size
            cell_size_via_meta = example_raster.meta.cell_size
            cell_size_via_raster_meta = example_raster.raster_meta.cell_size

            # Assert
            assert cell_size_via_property is cell_size_via_meta
            assert cell_size_via_property is cell_size_via_raster_meta
            assert cell_size_via_property == cell_size_via_meta
            assert cell_size_via_property == cell_size_via_raster_meta
            assert isinstance(cell_size_via_property, float)

        def test_cell_size_setter(self, example_raster: Raster):
            # Arrange
            new_cell_size = 5.0
            original_cell_size = example_raster.cell_size

            # Act
            example_raster.cell_size = new_cell_size

            # Assert
            assert example_raster.cell_size == new_cell_size
            assert example_raster.meta.cell_size == new_cell_size
            assert example_raster.raster_meta.cell_size == new_cell_size
            assert example_raster.cell_size != original_cell_size

    class TestSample:
        def test_sample_nan_raise(self, example_raster: Raster):
            with pytest.raises(
                ValueError, match="NaN value found in input coordinates"
            ):
                example_raster.sample([(0, 0), (1, np.nan)], na_action="raise")

        def test_sample_nan_ignore(self, example_raster: Raster):
            np.testing.assert_array_equal(
                example_raster.sample(
                    [(0, 0), (2, 2), (2, np.nan)], na_action="ignore"
                ),
                [1.0, 4, np.nan],
            )

        def test_oob_query(self, example_raster: Raster):
            result = example_raster.sample([(-99.0, 92640.20)], na_action="raise")
            np.testing.assert_array_equal(result, np.array([np.nan]))

        def test_raster_meta_with_irrelevant_fields(self):
            with pytest.raises(ValidationError):
                RasterMeta(
                    cell_size=1.0,
                    crs=CRS.from_epsg(2193),
                    transform=Affine(2.0, 0.0, 0.0, 0.0, 2.0, 0.0),
                    irrelevant_field="irrelevant",  # type: ignore[reportCallIssue]
                )

        def test_short_circuit(self):
            # Arrange
            raster = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=RasterMeta(
                    cell_size=1.0,
                    crs=CRS.from_epsg(2193),
                    transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                ),
            )

            # Act
            result = raster.sample([], na_action="raise")

            # Assert
            assert len(result) == 0

        def test_ndarray_input(self, example_raster: Raster):
            # Arrange
            coords = np.array([[0, 0], [1, 1]])

            # Act
            result = example_raster.sample(coords, na_action="raise")

            # Assert
            np.testing.assert_array_equal(result, np.array([1.0, 1.0]))

        def test_shapely_points_input(self, example_raster: Raster):
            # Arrange
            points = [Point(0, 0), Point(2, 2)]

            # Act
            result = example_raster.sample(points, na_action="raise")

            # Assert
            np.testing.assert_array_equal(result, np.array([1.0, 4.0]))

        def test_single_shapely_point_input(self, example_raster: Raster):
            # Arrange
            point = Point(0, 0)

            # Act
            result = example_raster.sample(point, na_action="raise")

            # Assert
            np.testing.assert_array_equal(result, np.array(1.0), strict=True)

        def test_single_tuple_input(self, example_raster: Raster):
            # Arrange
            coord = (0, 0)

            # Act
            result = example_raster.sample(coord, na_action="raise")

            # Assert
            np.testing.assert_array_equal(result, np.array(1.0), strict=True)

    class TestBounds:
        def test_bounds(self, example_raster: Raster):
            assert example_raster.bounds == (0.0, 0.0, 4.0, 4.0)

        def test_bounds_neg_scaled(self, example_neg_scaled_raster: Raster):
            assert example_neg_scaled_raster.bounds == (0.0, -4.0, 4.0, 0.0)

    class TestAsGeoDataFrame:
        def test_as_geodataframe(self, example_raster: Raster):
            import geopandas as gpd

            raster_gdf = example_raster.as_geodataframe(name="ben")

            expected_polygons = {
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
                Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            }

            # Check that the result is a GeoDataFrame
            assert isinstance(raster_gdf, gpd.GeoDataFrame)

            assert "ben" in raster_gdf.columns, "The name column is missing"

            # Check the CRS is correctly set
            assert raster_gdf.crs == example_raster.raster_meta.crs

            # Check the geometry and value columns are correct
            match_found = [
                any(raster_gdf.geometry.apply(lambda x, poly=poly: x.equals(poly)))
                for poly in expected_polygons
            ]
            assert all(match_found), (
                "Not all expected polygons match the geometries in the GeoDataFrame"
            )

    class TestAdd:
        def test_basic(self):
            # Arrange
            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            raster2 = Raster(
                arr=np.array([[5, 6], [7, 8]]),
                raster_meta=raster_meta,
            )

            # Act
            result = raster1 + raster2

            # Assert
            np.testing.assert_array_equal(result.arr, np.array([[6, 8], [10, 12]]))

        def test_add_subclass_return_type(self):
            # Arrange
            class MyRaster(Raster):
                pass

            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = MyRaster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )
            raster2 = MyRaster(
                arr=np.array([[5, 6], [7, 8]]),
                raster_meta=raster_meta,
            )

            # Act
            result = raster1 + raster2

            # Assert
            assert isinstance(result, MyRaster)

        def test_crs_mismatch(self):
            # Arrange
            raster_meta1 = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta1,
            )

            raster_meta2 = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(4326),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster2 = Raster(
                arr=np.array([[5, 6], [7, 8]]),
                raster_meta=raster_meta2,
            )

            # Act
            with pytest.raises(ValueError, match="Rasters must have the same metadata"):
                raster1 + raster2

        def test_right_add_float(self):
            # Arrange
            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            # Act
            result = 1.0 + raster1

            # Assert
            np.testing.assert_array_equal(result.arr, np.array([[2, 3], [4, 5]]))

        def test_shape_mismatch(self):
            # Arrange
            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            raster2 = Raster(
                arr=np.array([[5, 6]]),
                raster_meta=raster_meta,
            )

            # Act
            with pytest.raises(ValueError, match="Rasters must have the same shape"):
                raster1 + raster2

        def test_add_string_fails(self):
            # Arrange
            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            # Act
            with pytest.raises(TypeError, match="unsupported operand type"):
                raster + "hello"  # type: ignore[reportOperatorIssue]

        def test_preserves_dtype_float32(self, float32_raster: Raster):
            """Test that addition with scalar preserves float32."""
            result = float32_raster + 1.0
            assert result.arr.dtype == np.float32

        def test_preserves_dtype_float64(self, float64_raster: Raster):
            """Test that addition with scalar preserves float64."""
            result = float64_raster + 1.0
            assert result.arr.dtype == np.float64

        def test_preserves_dtype_float16(self, float16_raster: Raster):
            """Test that addition with scalar preserves float16."""
            result = float16_raster + 1.0
            assert result.arr.dtype == np.float16

        def test_raster_addition_preserves_dtype(self, float32_raster: Raster):
            """Test that raster-to-raster addition preserves dtype."""
            other = float32_raster.model_copy()
            result = float32_raster + other
            assert result.arr.dtype == np.float32

    class TestMul:
        def test_basic(self):
            # Arrange
            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            raster2 = Raster(
                arr=np.array([[5, 6], [7, 8]]),
                raster_meta=raster_meta,
            )

            # Act
            result = raster1 * raster2

            # Assert
            np.testing.assert_array_equal(result.arr, np.array([[5, 12], [21, 32]]))

        def test_crs_mismatch(self):
            # Arrange
            raster_meta1 = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta1,
            )

            raster_meta2 = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(4326),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster2 = Raster(
                arr=np.array([[5, 6], [7, 8]]),
                raster_meta=raster_meta2,
            )

            # Act
            with pytest.raises(ValueError, match="Rasters must have the same metadata"):
                raster1 * raster2

        def test_right_mul_float(self):
            # Arrange
            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            # Act
            result = 2.0 * raster1

            # Assert
            np.testing.assert_array_equal(result.arr, np.array([[2, 4], [6, 8]]))

        def test_shape_mismatch(self):
            # Arrange
            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            raster2 = Raster(
                arr=np.array([[5, 6]]),
                raster_meta=raster_meta,
            )

            # Act
            with pytest.raises(ValueError, match="Rasters must have the same shape"):
                raster1 * raster2

        def test_mul_string_fails(self):
            # Arrange
            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            # Act
            with pytest.raises(TypeError):
                raster * "hello"  # type: ignore[reportOperatorIssue]

        def test_preserves_dtype_float32(self, float32_raster: Raster):
            """Test that multiplication preserves float32."""
            result = float32_raster * 2.0
            assert result.arr.dtype == np.float32

        def test_preserves_dtype_float64(self, float64_raster: Raster):
            """Test that multiplication preserves float64."""
            result = float64_raster * 2.0
            assert result.arr.dtype == np.float64

    class TestTrueDiv:
        def test_basic(self):
            # Arrange
            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            raster2 = Raster(
                arr=np.array([[5, 6], [7, 8]]),
                raster_meta=raster_meta,
            )

            # Act
            result = raster1 / raster2

            # Assert
            np.testing.assert_array_equal(
                result.arr, np.array([[1 / 5, 2 / 6], [3 / 7, 4 / 8]])
            )

        def test_crs_mismatch(self):
            # Arrange
            raster_meta1 = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta1,
            )

            raster_meta2 = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(4326),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster2 = Raster(
                arr=np.array([[5, 6], [7, 8]]),
                raster_meta=raster_meta2,
            )

            # Act
            with pytest.raises(ValueError, match="Rasters must have the same metadata"):
                raster1 / raster2

        def test_right_div_float(self):
            # Arrange
            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            # Act
            result = 2.0 / raster1

            # Assert
            np.testing.assert_array_equal(result.arr, np.array([[0.5, 1], [1.5, 2]]))

        def test_shape_mismatch(self):
            # Arrange
            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            raster2 = Raster(
                arr=np.array([[5, 6]]),
                raster_meta=raster_meta,
            )

            # Act
            with pytest.raises(ValueError, match="Rasters must have the same shape"):
                raster1 / raster2

        def test_div_string_fails(self):
            # Arrange
            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            # Act
            with pytest.raises(TypeError):
                raster / "hello"  # type: ignore[reportOperatorIssue]

        def test_preserves_dtype_float32(self, float32_raster: Raster):
            """Test that division preserves float32."""
            result = float32_raster / 2.0
            assert result.arr.dtype == np.float32

    class TestSub:
        def test_basic(self):
            # Arrange
            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            raster2 = Raster(
                arr=np.array([[5, 6], [7, 8]]),
                raster_meta=raster_meta,
            )

            # Act
            result = raster1 - raster2

            # Assert
            np.testing.assert_array_equal(result.arr, np.array([[-4, -4], [-4, -4]]))

        def test_right_subtract_float(self):
            # Arrange
            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = Raster(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            # Act
            result = 1.0 - raster1

            # Assert
            np.testing.assert_array_equal(result.arr, np.array([[0, -1], [-2, -3]]))

        def test_preserves_dtype_float32(self, float32_raster: Raster):
            """Test that subtraction preserves float32."""
            result = float32_raster - 1.0
            assert result.arr.dtype == np.float32

        def test_negation_preserves_dtype_float32(self, float32_raster: Raster):
            """Test that negation preserves float32."""
            result = -float32_raster
            assert result.arr.dtype == np.float32

    class TestApply:
        def test_sine(self, example_raster: Raster):
            # Act
            result = example_raster.apply(np.sin)

            # Assert
            np.testing.assert_array_equal(result.arr, np.sin(example_raster.arr))

        def test_preserves_dtype_float32(self, float32_raster: Raster):
            """Test that apply() preserves dtype."""
            result = float32_raster.apply(lambda x: x * 2)
            assert result.arr.dtype == np.float32

        def test_apply_raw_preserves_dtype_float32(self, float32_raster: Raster):
            """Test that apply() with raw=True preserves dtype."""
            result = float32_raster.apply(lambda arr: arr * 2, raw=True)
            assert result.arr.dtype == np.float32

    class TestToFile:
        def test_saving_gtiff(self, tmp_path: Path, example_raster: Raster):
            # Arrange
            filename = tmp_path / "test_raster.tif"

            # Act
            example_raster.to_file(filename)

            # Assert
            assert filename.exists()

        def test_saving_grd_file(self, tmp_path: Path, example_raster: Raster):
            # Arrange
            filename = tmp_path / "test_raster.grd"

            # Act
            example_raster.to_file(filename)

            # Assert
            assert filename.exists()

        def test_string_as_path(self, tmp_path: Path, example_raster: Raster):
            # Arrange
            filename = tmp_path / "test_raster.tif"

            # Act
            example_raster.to_file(filename.as_posix())

            # Assert
            assert filename.exists()

        def test_kwargs_passed_to_rasterio(
            self, tmp_path: Path, example_raster: Raster
        ):
            # Arrange
            filename = tmp_path / "test_raster.tif"

            # Act - pass compress kwarg to rasterio
            example_raster.to_file(filename, compress="lzw")

            # Assert
            assert filename.exists()
            with rasterio.open(filename) as src:
                assert src.compression.value.lower() == "lzw"

        def test_custom_nodata_value(self, tmp_path: Path):
            # Arrange
            meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0),
            )
            arr = np.array([[1.0, np.nan], [3.0, 4.0]])
            raster = Raster(arr=arr, raster_meta=meta)
            filename = tmp_path / "test_nodata.tif"

            # Act
            raster.to_file(filename, nodata=-9999.0)

            # Assert
            with rasterio.open(filename) as src:
                assert src.nodata == pytest.approx(-9999.0)
                read_arr = src.read(1)
                assert read_arr[0, 1] == pytest.approx(-9999.0)

        def test_nodata_replaces_nan_in_array(self, tmp_path: Path):
            # Arrange
            meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0),
            )
            arr = np.array([[1.0, np.nan], [np.nan, 4.0]])
            raster = Raster(arr=arr, raster_meta=meta)
            filename = tmp_path / "test_nodata_replace.tif"

            # Act
            raster.to_file(filename, nodata=-9999.0)

            # Assert
            with rasterio.open(filename) as src:
                read_arr = src.read(1)
                assert read_arr[0, 0] == pytest.approx(1.0)
                assert read_arr[0, 1] == pytest.approx(-9999.0)
                assert read_arr[1, 0] == pytest.approx(-9999.0)
                assert read_arr[1, 1] == pytest.approx(4.0)

        def test_default_nodata_is_nan(self, tmp_path: Path):
            # Arrange
            filename = tmp_path / "test_default_nodata.tif"
            meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0),
            )
            arr = np.array([[1.0, 2.0], [3.0, 4.0]])
            raster = Raster(arr=arr, raster_meta=meta)

            # Act
            raster.to_file(filename)

            # Assert
            with rasterio.open(filename) as src:
                assert np.isnan(src.nodata)

    class TestPlot:
        def test_cell_array_unchanged(self, example_raster_with_zeros: Raster):
            # Arrange
            original_array = example_raster_with_zeros.arr.copy()

            # Act
            # Suppression will modify a raster copy internally, but not the original
            example_raster_with_zeros.plot(suppressed=0)

            # Assert
            np.testing.assert_array_equal(example_raster_with_zeros.arr, original_array)

        def test_plot_without_matplotlib_raises(self, monkeypatch: pytest.MonkeyPatch):
            # Arrange a minimal raster
            arr = np.array([[1.0, 2.0], [3.0, 4.0]])
            meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
            )
            raster = Raster(arr=arr, raster_meta=meta)

            # Simulate matplotlib not installed
            monkeypatch.setattr(
                "rastr.raster.MATPLOTLIB_INSTALLED", False, raising=False
            )

            # Act / Assert
            with pytest.raises(ImportError, match=r"matplotlib.*required"):
                raster.plot()

        def test_suppress_zeros(self):
            # Arrange
            raster = Raster.example()
            raster.arr[raster.arr < 0.1] = 0

            # Act, Assert - just checking it runs without error
            raster.plot(suppressed=0)

        def test_suppress_multiple(self):
            # Arrange
            raster = Raster.example()
            raster.arr[raster.arr < 0.1] = 0
            raster.arr[raster.arr > 0.2] = 0.2

            # Act, Assert - just checking it runs without error
            raster.plot(suppressed=[0, 0.2])

        def test_suppress_mocked(self):
            """Check suppressed values don't get passed to rasterio.plot.show"""
            # Arrange
            raster = Raster.example()
            raster.arr[raster.arr < 0.1] = 0
            raster.arr[raster.arr > 0.2] = 0.2

            with patch("rastr.raster.Raster.rio_show", autospec=True) as mock_show:
                mock_show.return_value = [None]

                # Act
                raster.plot(suppressed=[0.0, 0.2])

                # Assert
                args, _kwargs = mock_show.call_args
                model = args[0]
                assert np.all(~np.isin(model.arr, [0.0, 0.2]))

        def test_no_suppress_mocked(self):
            """Check non-suppressed values do get passed to rasterio.plot.show"""
            # Arrange
            raster = Raster.example()
            raster.arr[raster.arr < 0.1] = 0.0
            raster.arr[raster.arr > 0.2] = 0.2

            with patch("rastr.raster.Raster.rio_show", autospec=True) as mock_show:
                mock_show.return_value = [None]

                # Act
                raster.plot()

                # Assert
                args, _kwargs = mock_show.call_args
                model = args[0]
                assert np.any(np.isin(model.arr, [0.0, 0.2]))

        def test_plot_with_alpha_kwargs(self, example_raster_with_zeros: Raster):
            import matplotlib.pyplot as plt

            # Arrange
            fig, ax = plt.subplots()

            # Act
            ax = example_raster_with_zeros.plot(alpha=0.5, ax=ax)

            # Assert
            assert ax is not None
            plt.close(fig)

        def test_plot_with_additional_kwargs(self, example_raster_with_zeros: Raster):
            import matplotlib.pyplot as plt

            # Arrange
            fig, ax = plt.subplots()

            # Act - passing a rasterio.plot.show parameter that should be accepted
            ax = example_raster_with_zeros.plot(
                alpha=0.7, interpolation="bilinear", ax=ax
            )

            # Assert
            assert ax is not None
            plt.close(fig)

    class TestExample:
        def test_example(self):
            # Act
            raster = Raster.example()

            # Assert
            assert isinstance(raster, Raster)

    class TestFullLike:
        def test_basic_usage(self, example_raster: Raster):
            # Act
            filled_raster = Raster.full_like(example_raster, fill_value=5.0)

            # Assert
            assert filled_raster.shape == example_raster.shape
            assert filled_raster.raster_meta == example_raster.raster_meta
            expected_arr = np.array([[5.0, 5.0], [5.0, 5.0]])
            np.testing.assert_array_equal(filled_raster.arr, expected_arr)

        def test_with_nan_fill(self, example_raster: Raster):
            # Act
            filled_raster = Raster.full_like(example_raster, fill_value=np.nan)

            # Assert
            assert filled_raster.shape == example_raster.shape
            assert filled_raster.raster_meta == example_raster.raster_meta
            assert np.all(np.isnan(filled_raster.arr))

        def test_with_zero_fill(self, example_raster: Raster):
            # Act
            filled_raster = Raster.full_like(example_raster, fill_value=0.0)

            # Assert
            assert filled_raster.shape == example_raster.shape
            assert filled_raster.raster_meta == example_raster.raster_meta
            expected_arr = np.array([[0.0, 0.0], [0.0, 0.0]])
            np.testing.assert_array_equal(filled_raster.arr, expected_arr)

        def test_different_size_raster(self):
            # Arrange
            meta = RasterMeta(
                cell_size=2.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(2.0, 0.0, 0.0, 0.0, 2.0, 0.0),
            )
            large_raster = Raster(
                arr=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                raster_meta=meta,
            )

            # Act
            filled_raster = Raster.full_like(large_raster, fill_value=42.0)

            # Assert
            assert filled_raster.shape == (3, 3)
            assert filled_raster.raster_meta == meta
            np.testing.assert_array_equal(
                filled_raster.arr,
                np.array([[42.0, 42.0, 42.0], [42.0, 42.0, 42.0], [42.0, 42.0, 42.0]]),
            )

    class TestReadFile:
        def test_basic_tif(self, assets_dir: Path):
            # Arrange
            raster_path = assets_dir / "pga_g_clipped.tif"

            # Act
            raster = Raster.read_file(raster_path)

            # Assert
            assert isinstance(raster, Raster)
            assert raster.arr.shape == (2, 2)
            assert raster.raster_meta.crs.to_epsg() == 4326

    class TestFillNA:
        def test_2by2_example(self):
            # Arrange
            raster = Raster(
                arr=np.array([[1, float("nan")], [np.nan, 4]]),
                raster_meta=RasterMeta(
                    cell_size=1.0,
                    crs=CRS.from_epsg(2193),
                    transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                ),
            )

            # Act
            filled_raster = raster.fillna(0)

            # Assert
            np.testing.assert_array_equal(filled_raster.arr, np.array([[1, 0], [0, 4]]))

    class TestCopy:
        def test_returns_new_instance(self, example_raster: Raster):
            # Act
            copied = example_raster.copy()

            # Assert
            assert isinstance(copied, Raster)
            assert copied is not example_raster

        def test_preserves_array_values(self, example_raster: Raster):
            # Act
            copied = example_raster.copy()

            # Assert
            np.testing.assert_array_equal(copied.arr, example_raster.arr)

        def test_preserves_metadata(self, example_raster: Raster):
            # Act
            copied = example_raster.copy()

            # Assert
            assert copied.raster_meta == example_raster.raster_meta

        def test_modifications_dont_affect_original(self, example_raster: Raster):
            # Act
            copied = example_raster.copy()
            copied.arr[0, 0] = 999.0

            # Assert
            assert example_raster.arr[0, 0] != 999.0

        def test_preserves_dtype_float32(self, float32_raster: Raster):
            """Test that fillna() preserves dtype."""
            raster_with_nan = float32_raster.model_copy()
            raster_with_nan.arr[0, 0] = np.nan
            result = raster_with_nan.fillna(0.0)
            assert result.arr.dtype == np.float32

        def test_preserves_dtype_float64(self, float64_raster: Raster):
            """Test that fillna() preserves dtype for float64."""
            raster_with_nan = float64_raster.model_copy()
            raster_with_nan.arr[0, 0] = np.nan
            result = raster_with_nan.fillna(0.0)
            assert result.arr.dtype == np.float64

        def test_preserves_dtype_float16(self, float16_raster: Raster):
            """Test that fillna() preserves dtype for float16."""
            raster_with_nan = float16_raster.model_copy()
            raster_with_nan.arr[0, 0] = np.nan
            result = raster_with_nan.fillna(0.0)
            assert result.arr.dtype == np.float16

    class TestGetXY:
        def test_get_xy(self, example_raster: Raster):
            # Act
            x, y = example_raster.get_xy()

            # Assert
            # N.B. the xy coordinates in meshgrid style - 2D arrays
            expected_x = np.array([[1.0, 3.0], [1.0, 3.0]])
            expected_y = np.array([[1.0, 1.0], [3.0, 3.0]])
            np.testing.assert_array_equal(x, expected_x)
            np.testing.assert_array_equal(y, expected_y)

    class TestBlur:
        def test_numeric_propertoes(self, example_raster: Raster):
            # Act
            blurred_raster = example_raster.blur(sigma=1.0)

            # Assert
            assert isinstance(blurred_raster, Raster)

            # Standard deviation
            original_std = np.std(example_raster.arr)
            blurred_std = np.std(blurred_raster.arr)
            assert blurred_std < original_std, (
                "Standard deviation of blurred raster should be less than original."
                "This is because the blurring process reduces the variability in the"
                "data."
            )

            # Mean
            original_mean = np.mean(example_raster.arr)
            blurred_mean = np.mean(blurred_raster.arr)
            (
                pytest.approx(original_mean) == blurred_mean,
                ("Mean of blurred raster should be close to original mean"),
            )

        def test_preserves_dtype_float32(self, float32_raster: Raster):
            """Test that blur() preserves dtype."""
            result = float32_raster.blur(sigma=0.5)
            assert result.arr.dtype == np.float32

        def test_preserves_dtype_float64(self, float64_raster: Raster):
            """Test that blur() preserves dtype for float64."""
            result = float64_raster.blur(sigma=0.5)
            assert result.arr.dtype == np.float64

        def test_preserve_nan_preserves_nan_mask(self):
            # Arrange
            meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            arr = np.array(
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, 1.0, 2.0, 3.0, np.nan],
                    [np.nan, 4.0, 5.0, 6.0, np.nan],
                    [np.nan, 7.0, 8.0, 9.0, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                dtype=float,
            )
            raster = Raster(arr=arr, raster_meta=meta)
            original_nan_mask = np.isnan(raster.arr)

            # Act
            blurred = raster.blur(sigma=0.5, preserve_nan=True)

            # Assert
            assert np.array_equal(np.isnan(blurred.arr), original_nan_mask)

        def test_preserve_nan_blurs_valid_values(self):
            # Arrange
            meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            arr = np.array(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, 5.0, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                dtype=float,
            )
            raster = Raster(arr=arr, raster_meta=meta)

            # Act
            blurred = raster.blur(sigma=0.5, preserve_nan=True)

            # Assert
            assert not np.isnan(blurred.arr[1, 1])

        def test_preserve_nan_without_nans_behaves_normally(self):
            # Arrange
            meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
            raster = Raster(arr=arr, raster_meta=meta)

            # Act
            blurred_default = raster.blur(sigma=0.5)
            blurred_preserve = raster.blur(sigma=0.5, preserve_nan=True)

            # Assert
            np.testing.assert_array_almost_equal(
                blurred_default.arr, blurred_preserve.arr
            )

        def test_default_preserve_nan_is_true(self):
            # Arrange
            meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            arr = np.array(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, 5.0, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                dtype=float,
            )
            raster = Raster(arr=arr, raster_meta=meta)
            original_nan_mask = np.isnan(raster.arr)

            # Act
            blurred = raster.blur(sigma=0.5)

            # Assert - default behavior should preserve NaNs
            assert np.array_equal(np.isnan(blurred.arr), original_nan_mask)
            assert not np.isnan(blurred.arr[1, 1])

        def test_preserve_nan_false_spreads_nans(self):
            # Arrange
            meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            arr = np.array(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, 5.0, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                dtype=float,
            )
            raster = Raster(arr=arr, raster_meta=meta)

            # Act
            blurred = raster.blur(sigma=0.5, preserve_nan=False)

            # Assert - NaNs should spread into data
            assert np.all(np.isnan(blurred.arr))

    class TestExtrapolate:
        class TestNearest:
            def test_no_nas_stays_the_same(self, example_raster: Raster):
                # Act
                extrapolated_raster = example_raster.extrapolate(method="nearest")

                # Assert
                assert isinstance(extrapolated_raster, Raster)
                np.testing.assert_array_equal(
                    extrapolated_raster.arr, example_raster.arr
                )

            def test_fillna(self, example_raster: Raster):
                # Arrange
                raster_with_nas = example_raster
                raster_with_nas.arr[0, 0] = np.nan

                # Act
                extrapolated_raster = raster_with_nas.extrapolate(method="nearest")

                # Assert
                assert isinstance(extrapolated_raster, Raster)
                np.testing.assert_array_equal(
                    extrapolated_raster.arr,
                    np.array(
                        [[2, 2], [3, 4]]
                    ),  # NaN should be filled with nearest value
                )

            def test_start_with_all_na(self):
                # Arrange
                raster = Raster(
                    arr=np.array([[np.nan, np.nan], [np.nan, np.nan]]),
                    raster_meta=RasterMeta(
                        cell_size=1.0,
                        crs=CRS.from_epsg(2193),
                        transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                    ),
                )

                # Act
                extrapolated_raster = raster.extrapolate(method="nearest")

                # Assert
                assert isinstance(extrapolated_raster, Raster)
                np.testing.assert_array_equal(
                    extrapolated_raster.arr,
                    np.array(
                        [[np.nan, np.nan], [np.nan, np.nan]]
                    ),  # No change expected
                )

            def test_preserves_dtype_float32(self, float32_raster: Raster):
                """Test that extrapolate() preserves dtype."""
                raster_with_nan = float32_raster.model_copy()
                raster_with_nan.arr[0, 0] = np.nan
                result = raster_with_nan.extrapolate()
                assert result.arr.dtype == np.float32

            def test_preserves_dtype_float64(self, float64_raster: Raster):
                """Test that extrapolate() preserves dtype for float64."""
                raster_with_nan = float64_raster.model_copy()
                raster_with_nan.arr[0, 0] = np.nan
                result = raster_with_nan.extrapolate()
                assert result.arr.dtype == np.float64

            def test_preserves_dtype_float16(self, float16_raster: Raster):
                """Test that extrapolate() preserves dtype for float16."""
                raster_with_nan = float16_raster.model_copy()
                raster_with_nan.arr[0, 0] = np.nan
                result = raster_with_nan.extrapolate()
                assert result.arr.dtype == np.float16

    class TestContour:
        def test_contour_with_list_levels(self):
            import geopandas as gpd

            # Arrange
            raster = Raster.example()
            levels = [0.0, 0.5]

            # Act
            contour_gdf = raster.contour(levels=levels)

            # Assert
            assert isinstance(contour_gdf, gpd.GeoDataFrame)
            assert "level" in contour_gdf.columns
            assert len(contour_gdf) >= 0  # Should return some contours or empty GDF

        def test_contour_with_ndarray_levels(self):
            import geopandas as gpd

            # Arrange
            raster = Raster.example()
            levels = np.array([0.0, 0.5])

            # Act
            contour_gdf = raster.contour(levels=levels)

            # Assert
            assert isinstance(contour_gdf, gpd.GeoDataFrame)
            assert "level" in contour_gdf.columns
            assert len(contour_gdf) >= 0  # Should return some contours or empty GDF

        def test_contour_list_and_ndarray_equivalent(self):
            # Arrange
            raster = Raster.example()
            levels_list = [0.0, 0.5]
            levels_array = np.array([0.0, 0.5])

            # Act
            contour_gdf_list = raster.contour(levels=levels_list)
            contour_gdf_array = raster.contour(levels=levels_array)

            # Assert
            # Results should be equivalent (same number of contours at same levels)
            assert len(contour_gdf_list) == len(contour_gdf_array)
            assert list(contour_gdf_list["level"]) == list(contour_gdf_array["level"])

        def test_contour_positional_levels(self):
            # Arrange
            raster = Raster.example()
            levels = [0.0, 0.5]

            # Act - should pass without error when using positional levels arg
            contour_gdf = raster.contour(levels)  # noqa: F841

        def test_contour_returns_gdf_with_correct_columns(self):
            import geopandas as gpd

            raster = Raster.example()
            gdf = raster.contour(levels=[0.0, 0.5])

            assert isinstance(gdf, gpd.GeoDataFrame)
            assert list(gdf.columns) == ["level", "geometry"]
            assert "level" in gdf.columns
            assert "geometry" in gdf.columns

        def test_contour_levels_in_result(self):
            raster = Raster.example()
            levels = [0.0, 0.5]
            gdf = raster.contour(levels=levels)

            result_levels = set(gdf["level"].unique())
            expected_levels = set(levels)
            assert result_levels == expected_levels

        def test_contour_dissolve_behavior_one_row_per_level(self):
            raster = Raster.example()
            levels = [0.0, 0.5]
            gdf = raster.contour(levels=levels)

            # After dissolving, should have exactly one row per level
            assert len(gdf) == len(levels)
            assert set(gdf["level"]) == set(levels)

            # Geometries should be MultiLineString (dissolved from multiple LineStrings)
            for geom in gdf.geometry:
                assert isinstance(
                    geom, (MultiLineString, LineString)
                )  # Can be either depending on dissolve result

        def test_contour_with_smoothing(self):
            raster = Raster.example()
            gdf = raster.contour(levels=[0.0], smoothing=True)

            assert len(gdf) > 0
            assert all(gdf["level"] == 0.0)

        def test_contour_without_smoothing(self):
            raster = Raster.example()
            gdf = raster.contour(levels=[0.0], smoothing=False)

            assert len(gdf) > 0
            assert all(gdf["level"] == 0.0)

        def test_level_at_max(self):
            # https://github.com/tonkintaylor/rastr/issues/154

            # Arrange
            raster = Raster(
                arr=np.array([[1, 4, 4, 2], [1, 2, 4, 2], [1, 2, 4, 2], [1, 2, 4, 2]]),
                meta=RasterMeta.example(),
            )

            # Act
            gdf = raster.contour(levels=[4])

            # Assert
            assert len(gdf) > 0
            assert set(gdf["level"]) == {4.0}

        def test_level_at_min(self):
            # Arrange
            raster = Raster(
                arr=np.array([[1, 4, 4, 2], [1, 2, 4, 2], [1, 2, 4, 2], [1, 2, 4, 2]]),
                meta=RasterMeta.example(),
            )

            # Act
            gdf = raster.contour(levels=[1])

            # Assert
            assert len(gdf) > 0
            assert set(gdf["level"]) == {1.0}

        def test_contour_with_tuple_levels(self):
            # Arrange
            raster = Raster.example()
            levels = (0.0, 0.5)

            # Act
            contour_gdf = raster.contour(levels=levels)

            # Assert
            result_levels = set(contour_gdf["level"].unique())
            expected_levels = set(levels)
            assert result_levels == expected_levels

        def test_contour_with_set_levels(self):
            # Arrange
            raster = Raster.example()
            levels = {0.0, 0.5}

            # Act
            contour_gdf = raster.contour(levels=levels)

            # Assert
            result_levels = set(contour_gdf["level"].unique())
            expected_levels = levels
            assert result_levels == expected_levels


@pytest.fixture
def base_raster():
    meta = RasterMeta(
        cell_size=10.0,  # 10-meter cells
        crs=CRS.from_epsg(2193),
        transform=Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0),  # Standard NZTM-like
    )
    # Create a 4x4 raster with values 1-16
    arr = np.arange(1, 17, dtype=float).reshape(4, 4)
    return Raster(arr=arr, raster_meta=meta)


@pytest.fixture
def small_raster():
    meta = RasterMeta(
        cell_size=5.0,
        crs=CRS.from_epsg(2193),
        transform=Affine(5.0, 0.0, 0.0, 0.0, -5.0, 10.0),
    )
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    return Raster(arr=arr, raster_meta=meta)


class TestCrop:
    def test_fully_within_bbox_base(self, base_raster: Raster):
        # Arrange
        bounds = base_raster.bounds

        # Act
        cropped = base_raster.crop(bounds)

        # Assert
        assert cropped == base_raster

    def test_fully_within_bbox_small(self, small_raster: Raster):
        # Arrange
        bounds = small_raster.bounds

        # Act
        cropped = small_raster.crop(bounds)

        # Assert
        assert cropped == small_raster

    def test_crop_y_only(self, base_raster: Raster):
        # Arrange
        minx, miny, maxx, maxy = base_raster.bounds
        cell_size = base_raster.raster_meta.cell_size
        bounds = (minx, miny + cell_size, maxx, maxy - cell_size)
        expected_transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0 - cell_size)

        # Act
        cropped = base_raster.crop(bounds)

        # Assert
        assert cropped.arr.shape == (2, 4)  # Y-crop reduces rows, keeps columns
        assert cropped.bounds == bounds
        assert cropped.raster_meta.cell_size == base_raster.raster_meta.cell_size
        assert cropped.raster_meta.crs == base_raster.raster_meta.crs
        assert cropped.raster_meta.transform == expected_transform

    def test_crop_x_only(self, base_raster: Raster):
        # Arrange
        minx, miny, maxx, maxy = base_raster.bounds
        cell_size = base_raster.raster_meta.cell_size
        bounds = (minx + cell_size, miny, maxx - cell_size, maxy)
        expected_transform = Affine(10.0, 0.0, minx + cell_size, 0.0, -10.0, 100.0)

        # Act
        cropped = base_raster.crop(bounds)

        # Assert
        assert cropped.arr.shape == (4, 2)  # X-crop reduces columns, keeps rows
        assert cropped.bounds == bounds
        assert cropped.raster_meta.cell_size == base_raster.raster_meta.cell_size
        assert cropped.raster_meta.crs == base_raster.raster_meta.crs
        assert cropped.raster_meta.transform == expected_transform

    def test_underflow_crops_border_cells(self, base_raster: Raster):
        # Arrange
        minx, miny, maxx, maxy = base_raster.bounds
        cell_size = base_raster.raster_meta.cell_size
        shift = base_raster.raster_meta.cell_size / 10  # Some cells overlap bounds
        bounds = (minx + shift, miny + shift, maxx - shift, maxy - shift)
        expected_transform = Affine(
            10.0, 0.0, minx + cell_size, 0.0, -10.0, 100.0 - cell_size
        )  # Cells overlapping bounds are clipped

        # Act
        cropped = base_raster.crop(bounds)

        # Assert
        assert cropped.arr.shape == (2, 2)
        assert cropped.raster_meta.cell_size == base_raster.raster_meta.cell_size
        assert cropped.raster_meta.crs == base_raster.raster_meta.crs
        assert cropped.raster_meta.transform == expected_transform

    def test_overflow_doesnt_crop(self, base_raster: Raster):
        # Arrange
        minx, miny, maxx, maxy = base_raster.bounds
        shift = base_raster.raster_meta.cell_size / 10  # Some cells overlap bounds
        bounds = (minx + shift, miny + shift, maxx - shift, maxy - shift)

        # Act
        cropped = base_raster.crop(bounds, strategy="overflow")

        # Assert
        assert cropped == base_raster  # Border cells are not clipped, despite overlap

    @pytest.mark.parametrize("strategy", ["overflow", "underflow"])
    def test_boundary_case(
        self, base_raster: Raster, strategy: Literal["overflow", "underflow"]
    ):
        # Arrange
        minx, miny, maxx, maxy = base_raster.bounds
        bounds = (minx, miny, maxx - (maxx - minx) / 4, maxy - (maxy - miny) / 4)
        expected_transform = Affine(
            10, 0.0, minx, 0.0, -10.0, maxy - (maxy - miny) / 4
        )  # Cells on the upper right are removed

        # Act
        cropped = base_raster.crop(bounds, strategy=strategy)

        # Assert
        assert cropped.arr.shape == (3, 3)  # Should crop one side only
        assert cropped.raster_meta.transform == expected_transform
        assert cropped.bounds == bounds
        assert cropped.raster_meta.cell_size == base_raster.raster_meta.cell_size
        assert cropped.raster_meta.crs == base_raster.raster_meta.crs

    def test_overflow_crops(self, base_raster: Raster):
        # Arrange
        minx, miny, maxx, maxy = base_raster.bounds
        bounds = (minx + 11, miny + 11, maxx - 11, maxy - 11)
        expected_transform = Affine(10.0, 0.0, minx + 10, 0.0, -10.0, maxy - 10)

        # Act
        cropped = base_raster.crop(bounds, strategy="overflow")

        # Assert
        assert cropped.arr.shape == (2, 2)  # Cells on both sides are removed
        assert cropped.raster_meta.transform == expected_transform
        assert cropped.bounds == (minx + 10, miny + 10, maxx - 10, maxy - 10)
        assert cropped.raster_meta.cell_size == base_raster.raster_meta.cell_size
        assert cropped.raster_meta.crs == base_raster.raster_meta.crs

    @pytest.mark.parametrize(
        "bounds",
        [(1000, 1000, 2000, 2000), (0.0, 60.0, 0.0, 100.0)],
        ids=["out_of_bounds", "fully_clipped_x"],
    )
    def test_no_contained_data_raises(
        self, base_raster: Raster, bounds: tuple[float, float, float, float]
    ):
        # Arrange, Act & Assert
        with pytest.raises(
            ValueError,
            match=r"Cropped array is empty; no cells within the specified bounds\.",
        ):
            base_raster.crop(bounds)

    def test_crop_non_square_raster_indexing(self):
        """Test that crop method correctly indexes non-square rasters.

        This tests the fix for issue #140 where array indexing was backwards,
        causing spatial misalignment in cropped rasters.
        """
        # Arrange: Create a non-square raster with distinctive values
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0),
        )
        arr = np.array(
            [
                [1, 2, 3, 4, 5],  # row 0
                [6, 7, 8, 9, 10],  # row 1
                [11, 12, 13, 14, 15],  # row 2
            ],
            dtype=float,
        )
        raster = Raster(arr=arr, raster_meta=meta)

        # Act: Crop to select middle 3 columns (keeping all rows)
        bounds = (1.0, 0.0, 4.0, 3.0)  # Should select columns at x=1.5, 2.5, 3.5
        cropped = raster.crop(bounds)

        # Assert: Result should have all 3 rows but only 3 columns
        expected_shape = (3, 3)
        expected_array = np.array(
            [
                [2, 3, 4],  # row 0, columns 1,2,3 (0-indexed)
                [7, 8, 9],  # row 1, columns 1,2,3
                [12, 13, 14],  # row 2, columns 1,2,3
            ],
            dtype=float,
        )

        assert cropped.arr.shape == expected_shape
        np.testing.assert_array_equal(cropped.arr, expected_array)

    def test_unsupported_crop_strategy(self, base_raster: Raster):
        # Arrange
        bounds = base_raster.bounds

        # Act & Assert
        with pytest.raises(
            NotImplementedError,
            match="Unsupported cropping strategy: invalid_strategy",
        ):
            base_raster.crop(bounds, strategy="invalid_strategy")  # type: ignore[reportArgumentType]

    def test_strategy_is_keyword_only(self, base_raster: Raster):
        # Arrange
        bounds = base_raster.bounds

        # Act & Assert
        with pytest.raises(TypeError):
            base_raster.crop(bounds, "overflow")  # type: ignore[reportCallIssue]

    def test_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that crop() preserves dtype."""
        bounds = (0.0, 0.0, 1.5, 1.5)
        result = float32_raster.crop(bounds)
        assert result.arr.dtype == np.float32


class TestPad:
    def test_example(self):
        # Arrange
        raster = Raster(
            arr=np.array(
                [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25],
                ],
                dtype=float,
            ),
            raster_meta=RasterMeta.example(),
        )

        # Act
        width = 2.0  # 2 units in CRS coordinates
        padded = raster.pad(width=width)

        # Assert
        assert isinstance(padded, Raster)
        # Should pad by 1 cell on each side (2.0 / 2.0 = 1.0, ceil(1.0) = 1)
        assert padded.arr.shape == (7, 7)  # 5x5 + 2 padding on each side

        # Check that original data is in the center
        np.testing.assert_array_equal(padded.arr[1:6, 1:6], raster.arr)

        # Check that padding is NaN by default
        assert np.isnan(padded.arr[0, :]).all()  # Top row
        assert np.isnan(padded.arr[-1, :]).all()  # Bottom row
        assert np.isnan(padded.arr[:, 0]).all()  # Left column
        assert np.isnan(padded.arr[:, -1]).all()  # Right column

    def test_pad_with_custom_value(self):
        # Arrange
        raster = Raster(
            arr=np.array([[1, 2], [3, 4]], dtype=float),
            raster_meta=RasterMeta.example(),
        )

        # Act
        width = 2.0  # Same as cell size, so 1 cell padding
        fill_value = -999.0
        padded = raster.pad(width=width, value=fill_value)

        # Assert
        assert padded.arr.shape == (4, 4)  # 2x2 + 2 padding on each side

        # Check that padding uses custom value
        assert (padded.arr[0, :] == fill_value).all()  # Top row
        assert (padded.arr[-1, :] == fill_value).all()  # Bottom row
        assert (padded.arr[:, 0] == fill_value).all()  # Left column
        assert (padded.arr[:, -1] == fill_value).all()  # Right column

        # Check original data is preserved
        np.testing.assert_array_equal(padded.arr[1:3, 1:3], raster.arr)

    def test_pad_fractional_width(self):
        # Arrange
        raster = Raster(
            arr=np.array([[1, 2], [3, 4]], dtype=float),
            raster_meta=RasterMeta.example(),
        )

        # Act - use fractional width that should still result in 1 cell padding
        width = 1.5  # Less than cell size (2.0), but ceil(1.5/2.0) = ceil(0.75) = 1
        padded = raster.pad(width=width)

        # Assert
        assert padded.arr.shape == (4, 4)  # Still 1 cell padding on each side

    def test_pad_preserves_metadata(self):
        # Arrange
        raster = Raster(
            arr=np.array([[1, 2], [3, 4]], dtype=float),
            raster_meta=RasterMeta.example(),
        )

        # Act
        padded = raster.pad(width=4.0)  # 2 cells padding

        # Assert
        assert padded.raster_meta.cell_size == raster.raster_meta.cell_size
        assert padded.raster_meta.crs == raster.raster_meta.crs

        # Check bounds are expanded correctly
        orig_xmin, orig_ymin, orig_xmax, orig_ymax = raster.bounds
        new_xmin, new_ymin, new_xmax, new_ymax = padded.bounds

        expected_padding = 4.0  # 2 cells * 2.0 cell_size
        assert new_xmin == pytest.approx(orig_xmin - expected_padding)
        assert new_ymin == pytest.approx(orig_ymin - expected_padding)
        assert new_xmax == pytest.approx(orig_xmax + expected_padding)
        assert new_ymax == pytest.approx(orig_ymax + expected_padding)

    def test_pad_zero_width(self):
        # Arrange
        raster = Raster(
            arr=np.array([[1, 2], [3, 4]], dtype=float),
            raster_meta=RasterMeta.example(),
        )

        # Act
        padded = raster.pad(width=0.0)

        # Assert - should be unchanged when width is 0
        assert padded.arr.shape == raster.arr.shape
        np.testing.assert_array_equal(padded.arr, raster.arr)

    def test_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that pad() preserves dtype."""
        result = float32_raster.pad(width=1.0)
        assert result.arr.dtype == np.float32


class TestTaperBorder:
    def test_example(self):
        # Arrange
        raster = Raster(
            arr=np.array(
                [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25],
                ],
                dtype=float,
            ),
            raster_meta=RasterMeta.example(),
        )

        # Act
        w = 2.5
        s = raster.raster_meta.cell_size
        f = w / s
        softened = raster.taper_border(width=w)

        # Assert
        assert isinstance(softened, Raster)
        np.testing.assert_allclose(
            softened.arr,
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 7 / f, 8 / f, 9 / f, 0],
                    [0, 12 / f, 13, 14 / f, 0],
                    [0, 17 / f, 18 / f, 19 / f, 0],
                    [0, 0, 0, 0, 0],
                ],
            ),
        )

    def test_nonzero_limits(self):
        # Arrange
        raster = Raster.example()

        # Act
        softened = raster.taper_border(width=15.0, limit=20.0)

        # Assert
        assert isinstance(softened, Raster)
        # Check that values around the edges equal the limit
        assert np.all(softened.arr[0, :] == 20.0)
        assert np.all(softened.arr[-1, :] == 20.0)
        assert np.all(softened.arr[:, 0] == 20.0)
        assert np.all(softened.arr[:, -1] == 20.0)

    def test_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that taper_border() preserves dtype."""
        result = float32_raster.taper_border(width=0.5)
        assert result.arr.dtype == np.float32


class TestClip:
    def test_example(self):
        # Arrange
        raster = Raster(
            arr=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            raster_meta=RasterMeta.example(),
        )
        polygon = raster.bbox.buffer(-2.5)

        # Act
        clipped = raster.clip(polygon)

        # Assert
        assert isinstance(clipped, Raster)
        assert clipped.raster_meta == raster.raster_meta
        np.testing.assert_array_equal(
            clipped.arr,
            np.array(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, 5, np.nan],
                    [np.nan, np.nan, np.nan],
                ]
            ),
        )

    def test_own_bbox(self, base_raster: Raster):
        # Arrange
        polygon = base_raster.bbox

        # Act
        clipped = base_raster.clip(polygon)

        # Assert
        assert clipped == base_raster

    def test_multipolygon(self, base_raster: Raster):
        # Arrange
        minx, miny, maxx, maxy = base_raster.bounds
        cell_size = base_raster.raster_meta.cell_size
        poly1 = box(
            minx + cell_size, miny + cell_size, maxx - cell_size, maxy - cell_size
        )
        poly2 = box(minx, miny, minx + 2 * cell_size, miny + 2 * cell_size)
        multipoly = MultiPolygon([poly1, poly2])

        # Act
        clipped = base_raster.clip(multipoly)

        # Assert
        expected_array = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, 6, 7, np.nan],
                [9, 10, 11, np.nan],
                [13, 14, np.nan, np.nan],
            ]
        )
        np.testing.assert_array_equal(clipped.arr, expected_array)


class TestTrimNaN:
    def test_no_nan_values_unchanged(self, base_raster: Raster):
        # Arrange - base_raster has no NaN values

        # Act
        cropped = base_raster.trim_nan()

        # Assert
        assert cropped == base_raster
        assert cropped.arr.shape == base_raster.arr.shape
        np.testing.assert_array_equal(cropped.arr, base_raster.arr)
        assert cropped.raster_meta == base_raster.raster_meta

    def test_nan_edges_all_sides(self):
        # Arrange
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 5.0),
        )
        # Create 5x5 array with NaN border and 3x3 data center
        arr = np.full((5, 5), np.nan)
        arr[1:4, 1:4] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_nan()

        # Assert
        expected_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        np.testing.assert_array_equal(cropped.arr, expected_arr)
        assert cropped.arr.shape == (3, 3)

        # Check that bounds are correctly adjusted
        expected_transform = Affine(1.0, 0.0, 1.0, 0.0, -1.0, 4.0)
        assert cropped.raster_meta.transform == expected_transform

    def test_nan_top_bottom_only(self):
        # Arrange
        meta = RasterMeta(
            cell_size=2.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(2.0, 0.0, 0.0, 0.0, -2.0, 8.0),
        )
        # Create 4x3 array with NaN top and bottom rows
        arr = np.array(
            [
                [np.nan, np.nan, np.nan],
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [np.nan, np.nan, np.nan],
            ]
        )
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_nan()

        # Assert
        expected_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(cropped.arr, expected_arr)
        assert cropped.arr.shape == (2, 3)

        # Check transform adjustment (y origin should move down by 1 row)
        expected_transform = Affine(2.0, 0.0, 0.0, 0.0, -2.0, 6.0)
        assert cropped.raster_meta.transform == expected_transform

    def test_nan_left_right_only(self):
        # Arrange
        meta = RasterMeta(
            cell_size=1.5,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.5, 0.0, 0.0, 0.0, -1.5, 6.0),
        )
        # Create 3x4 array with NaN left and right columns
        arr = np.array(
            [
                [np.nan, 1.0, 2.0, np.nan],
                [np.nan, 3.0, 4.0, np.nan],
                [np.nan, 5.0, 6.0, np.nan],
            ]
        )
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_nan()

        # Assert
        expected_arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_equal(cropped.arr, expected_arr)
        assert cropped.arr.shape == (3, 2)

        # Check transform adjustment (x origin should move right by 1 column)
        expected_transform = Affine(1.5, 0.0, 1.5, 0.0, -1.5, 6.0)
        assert cropped.raster_meta.transform == expected_transform

    def test_asymmetric_nan_borders(self):
        # Arrange
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 6.0),
        )
        # Create 6x5 array with asymmetric NaN borders
        arr = np.full((6, 5), np.nan)
        # Data in a 2x2 region offset from center
        arr[2:4, 1:3] = np.array([[1.0, 2.0], [3.0, 4.0]])
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_nan()

        # Assert
        expected_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(cropped.arr, expected_arr)
        assert cropped.arr.shape == (2, 2)

        # Check transform adjustment
        expected_transform = Affine(1.0, 0.0, 1.0, 0.0, -1.0, 4.0)
        assert cropped.raster_meta.transform == expected_transform

    def test_single_non_nan_cell(self):
        # Arrange
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        )
        # Create 4x4 array with single non-NaN value
        arr = np.full((4, 4), np.nan)
        arr[1, 2] = 42.0
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_nan()

        # Assert
        expected_arr = np.array([[42.0]])
        np.testing.assert_array_equal(cropped.arr, expected_arr)
        assert cropped.arr.shape == (1, 1)

        # Check transform adjustment
        expected_transform = Affine(1.0, 0.0, 2.0, 0.0, -1.0, 3.0)
        assert cropped.raster_meta.transform == expected_transform

    def test_all_nan_raises_error(self):
        # Arrange
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0),
        )
        arr = np.full((3, 3), np.nan)
        raster = Raster(arr=arr, raster_meta=meta)

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot crop raster: all values are NaN"):
            raster.trim_nan()

    def test_mixed_nan_and_finite_values(self):
        # Arrange
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        )
        arr = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, 1.0, np.inf, np.nan],
                [np.nan, -np.inf, 2.0, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        )
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_nan()

        # Assert
        expected_arr = np.array([[1.0, np.inf], [-np.inf, 2.0]])
        np.testing.assert_array_equal(cropped.arr, expected_arr)
        assert cropped.arr.shape == (2, 2)

    def test_preserve_metadata(self):
        # Arrange
        original_crs = CRS.from_epsg(4326)  # Different CRS
        original_cell_size = 0.5
        meta = RasterMeta(
            cell_size=original_cell_size,
            crs=original_crs,
            transform=Affine(0.5, 0.0, 0.0, 0.0, -0.5, 2.0),
        )
        arr = np.array([[np.nan, np.nan], [1.0, 2.0]])
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_nan()

        # Assert
        assert cropped.raster_meta.crs == original_crs
        assert cropped.raster_meta.cell_size == original_cell_size
        # Only transform should change
        assert cropped.raster_meta.transform != raster.raster_meta.transform

    def test_return_type_subclass(self):
        # Arrange
        class MyRaster(Raster):
            pass

        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0),
        )
        arr = np.array(
            [
                [np.nan, np.nan, np.nan],
                [np.nan, 1.0, np.nan],
                [np.nan, np.nan, np.nan],
            ]
        )
        raster = MyRaster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_nan()

        # Assert
        assert isinstance(cropped, MyRaster)

    def test_original_raster_unchanged(self):
        # Arrange
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0),
        )
        original_arr = np.array(
            [
                [np.nan, np.nan, np.nan],
                [np.nan, 1.0, np.nan],
                [np.nan, np.nan, np.nan],
            ]
        )
        raster = Raster(arr=original_arr.copy(), raster_meta=meta)

        # Act
        cropped = raster.trim_nan()

        # Assert
        np.testing.assert_array_equal(raster.arr, original_arr)
        assert raster.raster_meta == meta
        assert cropped is not raster  # Different objects

    def test_complex_transform_preservation(self):
        # Arrange - create a transform with rotation/skew
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.1, 10.0, 0.1, -1.0, 20.0),  # Has rotation/skew
        )
        # Create array where we crop both rows and columns
        arr = np.array(
            [[np.nan, np.nan, np.nan], [np.nan, 1.0, 2.0], [np.nan, 3.0, 4.0]]
        )
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_nan()

        # Assert
        # The a, b, d, e components should be preserved
        original_transform = raster.raster_meta.transform
        new_transform = cropped.raster_meta.transform

        assert new_transform.a == original_transform.a  # x pixel size
        assert new_transform.b == original_transform.b  # row rotation
        assert new_transform.d == original_transform.d  # column rotation
        assert new_transform.e == original_transform.e  # y pixel size
        # Both c and f (origin) should change due to cropping
        assert new_transform.c != original_transform.c
        assert new_transform.f != original_transform.f

    def test_disconnected_data_regions(self):
        # Arrange
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 6.0),
        )
        # Create array with two disconnected data regions
        arr = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 1.0, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, 2.0, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
            ]
        )
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_nan()

        # Assert
        # Should crop to the bounding box that contains both data points
        expected_arr = np.array(
            [[1.0, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, 2.0]]
        )
        np.testing.assert_array_equal(cropped.arr, expected_arr)
        assert cropped.arr.shape == (3, 3)

        # Check transform adjustment (should move to include both data points)
        expected_transform = Affine(1.0, 0.0, 1.0, 0.0, -1.0, 5.0)
        assert cropped.raster_meta.transform == expected_transform

    def test_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that trim_nan() preserves dtype."""
        raster_with_nan = float32_raster.pad(width=1.0, value=np.nan)
        result = raster_with_nan.trim_nan()
        assert result.arr.dtype == np.float32


class TestTrimZeros:
    def test_no_zero_values_unchanged(self, base_raster: Raster):
        # Arrange - base_raster has no zero values

        # Act
        cropped = base_raster.trim_zeros()

        # Assert
        assert cropped == base_raster
        assert cropped.arr.shape == base_raster.arr.shape
        np.testing.assert_array_equal(cropped.arr, base_raster.arr)
        assert cropped.raster_meta == base_raster.raster_meta

    def test_zero_edges_all_sides(self):
        # Arrange
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 5.0),
        )
        # Create 5x5 array with zero border and 3x3 data center
        arr = np.zeros((5, 5))
        arr[1:4, 1:4] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_zeros()

        # Assert
        expected_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        np.testing.assert_array_equal(cropped.arr, expected_arr)
        assert cropped.arr.shape == (3, 3)

        # Check that bounds are correctly adjusted
        expected_transform = Affine(1.0, 0.0, 1.0, 0.0, -1.0, 4.0)
        assert cropped.raster_meta.transform == expected_transform

    def test_zero_top_bottom_only(self):
        # Arrange
        meta = RasterMeta(
            cell_size=2.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(2.0, 0.0, 0.0, 0.0, -2.0, 8.0),
        )
        # Create 4x3 array with zero top and bottom rows
        arr = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [0.0, 0.0, 0.0],
            ]
        )
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_zeros()

        # Assert
        expected_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(cropped.arr, expected_arr)
        assert cropped.arr.shape == (2, 3)

        # Check transform adjustment (y origin should move down by 1 row)
        expected_transform = Affine(2.0, 0.0, 0.0, 0.0, -2.0, 6.0)
        assert cropped.raster_meta.transform == expected_transform

    def test_zero_left_right_only(self):
        # Arrange
        meta = RasterMeta(
            cell_size=1.5,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.5, 0.0, 0.0, 0.0, -1.5, 6.0),
        )
        # Create 3x4 array with zero left and right columns
        arr = np.array(
            [
                [0.0, 1.0, 2.0, 0.0],
                [0.0, 3.0, 4.0, 0.0],
                [0.0, 5.0, 6.0, 0.0],
            ]
        )
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_zeros()

        # Assert
        expected_arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_equal(cropped.arr, expected_arr)
        assert cropped.arr.shape == (3, 2)

        # Check transform adjustment (x origin should move right by 1 column)
        expected_transform = Affine(1.5, 0.0, 1.5, 0.0, -1.5, 6.0)
        assert cropped.raster_meta.transform == expected_transform

    def test_asymmetric_zero_borders(self):
        # Arrange
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 6.0),
        )
        # Create 6x5 array with asymmetric zero borders
        arr = np.zeros((6, 5))
        # Data in a 2x2 region offset from center
        arr[2:4, 1:3] = np.array([[1.0, 2.0], [3.0, 4.0]])
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_zeros()

        # Assert
        expected_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(cropped.arr, expected_arr)
        assert cropped.arr.shape == (2, 2)

        # Check transform adjustment
        expected_transform = Affine(1.0, 0.0, 1.0, 0.0, -1.0, 4.0)
        assert cropped.raster_meta.transform == expected_transform

    def test_single_non_zero_cell(self):
        # Arrange
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        )
        # Create 4x4 array with single non-zero value
        arr = np.zeros((4, 4))
        arr[1, 2] = 42.0
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_zeros()

        # Assert
        expected_arr = np.array([[42.0]])
        np.testing.assert_array_equal(cropped.arr, expected_arr)
        assert cropped.arr.shape == (1, 1)

        # Check transform adjustment
        expected_transform = Affine(1.0, 0.0, 2.0, 0.0, -1.0, 3.0)
        assert cropped.raster_meta.transform == expected_transform

    def test_all_zeros_raises_error(self):
        # Arrange
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0),
        )
        arr = np.zeros((3, 3))
        raster = Raster(arr=arr, raster_meta=meta)

        # Act & Assert
        with pytest.raises(ValueError, match="Cannot crop raster: all values are zero"):
            raster.trim_zeros()

    def test_preserve_metadata(self):
        # Arrange
        original_crs = CRS.from_epsg(4326)  # Different CRS
        original_cell_size = 0.5
        meta = RasterMeta(
            cell_size=original_cell_size,
            crs=original_crs,
            transform=Affine(0.5, 0.0, 0.0, 0.0, -0.5, 2.0),
        )
        arr = np.array([[0.0, 0.0], [1.0, 2.0]])
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_zeros()

        # Assert
        assert cropped.raster_meta.crs == original_crs
        assert cropped.raster_meta.cell_size == original_cell_size
        # Only transform should change
        assert cropped.raster_meta.transform != raster.raster_meta.transform

    def test_return_type_subclass(self):
        # Arrange
        class MyRaster(Raster):
            pass

        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0),
        )
        arr = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        raster = MyRaster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_zeros()

        # Assert
        assert isinstance(cropped, MyRaster)

    def test_original_raster_unchanged(self):
        # Arrange
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0),
        )
        original_arr = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        raster = Raster(arr=original_arr.copy(), raster_meta=meta)

        # Act
        cropped = raster.trim_zeros()

        # Assert
        np.testing.assert_array_equal(raster.arr, original_arr)
        assert raster.raster_meta == meta
        assert cropped is not raster  # Different objects

    def test_complex_transform_preservation(self):
        # Arrange - create a transform with rotation/skew
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.1, 10.0, 0.1, -1.0, 20.0),  # Has rotation/skew
        )
        # Create array where we crop both rows and columns
        arr = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 3.0, 4.0]])
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_zeros()

        # Assert
        # The a, b, d, e components should be preserved
        original_transform = raster.raster_meta.transform
        new_transform = cropped.raster_meta.transform

        assert new_transform.a == original_transform.a  # x pixel size
        assert new_transform.b == original_transform.b  # row rotation
        assert new_transform.d == original_transform.d  # column rotation
        assert new_transform.e == original_transform.e  # y pixel size
        # Both c and f (origin) should change due to cropping
        assert new_transform.c != original_transform.c
        assert new_transform.f != original_transform.f

    def test_disconnected_data_regions(self):
        # Arrange
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 6.0),
        )
        # Create array with two disconnected data regions
        arr = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        cropped = raster.trim_zeros()

        # Assert
        # Should crop to the bounding box that contains both data points
        expected_arr = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        np.testing.assert_array_equal(cropped.arr, expected_arr)
        assert cropped.arr.shape == (3, 3)

        # Check transform adjustment (should move to include both data points)
        expected_transform = Affine(1.0, 0.0, 1.0, 0.0, -1.0, 5.0)
        assert cropped.raster_meta.transform == expected_transform


class TestResample:
    def test_upsampling_doubles_resolution(self, base_raster: Raster):
        # Arrange
        new_cell_size = 5.0  # Half the original size (10.0)

        # Act
        resampled = base_raster.resample(new_cell_size)

        # Assert
        assert resampled.raster_meta.cell_size == new_cell_size
        # Should approximately double the dimensions (some discretization)
        assert resampled.arr.shape[0] >= 7  # At least 2x original (4)
        assert resampled.arr.shape[1] >= 7
        assert resampled.raster_meta.crs == base_raster.raster_meta.crs

    def test_downsampling_halves_resolution(self, base_raster: Raster):
        # Arrange
        new_cell_size = 20.0  # Double the original size (10.0)

        # Act
        resampled = base_raster.resample(new_cell_size)

        # Assert
        assert resampled.raster_meta.cell_size == new_cell_size
        # Should approximately halve the dimensions
        assert resampled.arr.shape[0] <= 3  # At most half original (4)
        assert resampled.arr.shape[1] <= 3
        assert resampled.raster_meta.crs == base_raster.raster_meta.crs

    def test_same_cell_size_returns_similar_raster(self, base_raster: Raster):
        # Arrange
        original_cell_size = base_raster.raster_meta.cell_size

        # Act
        resampled = base_raster.resample(original_cell_size)

        # Assert
        assert resampled.raster_meta.cell_size == original_cell_size
        # Dimensions should be the same or very close due to discretization
        assert abs(resampled.arr.shape[0] - base_raster.arr.shape[0]) <= 1
        assert abs(resampled.arr.shape[1] - base_raster.arr.shape[1]) <= 1

    def test_extreme_upsampling(self, small_raster: Raster):
        # Arrange
        new_cell_size = 1.0  # Much smaller than original 5.0

        # Act
        resampled = small_raster.resample(new_cell_size)

        # Assert
        assert resampled.raster_meta.cell_size == new_cell_size
        # Should be significantly larger
        assert resampled.arr.shape[0] >= 8
        assert resampled.arr.shape[1] >= 8

    def test_extreme_downsampling(self, base_raster: Raster):
        # Arrange
        new_cell_size = 100.0  # Much larger than original 10.0

        # Act
        resampled = base_raster.resample(new_cell_size)

        # Assert
        assert resampled.raster_meta.cell_size == new_cell_size
        # Should be much smaller, potentially 1x1
        assert resampled.arr.shape[0] >= 1
        assert resampled.arr.shape[1] >= 1
        assert resampled.arr.shape[0] <= 2
        assert resampled.arr.shape[1] <= 2

    def test_transform_scaling(self, small_raster: Raster):
        # Arrange
        new_cell_size = 2.5  # Half the original cell size

        # Act
        resampled = small_raster.resample(new_cell_size)

        # Assert
        new_transform = resampled.raster_meta.transform
        # The transform scale should be updated to reflect new cell size
        assert abs(abs(new_transform.a) - new_cell_size) < 0.1
        assert abs(abs(new_transform.e) - new_cell_size) < 0.1

    def test_bilinear_interpolation_smoothing(self, small_raster: Raster):
        # Arrange
        new_cell_size = 2.0  # Between original cells

        # Act
        resampled = small_raster.resample(new_cell_size)

        # Assert
        # With bilinear interpolation, we shouldn't have any extreme values
        # that are outside the range of the original data
        original_min = small_raster.min()
        original_max = small_raster.max()
        resampled_min = resampled.min()
        resampled_max = resampled.max()

        # Values should generally be within the original range
        # (allowing small numerical tolerances)
        assert resampled_min >= original_min - 0.1
        assert resampled_max <= original_max + 0.1

    def test_invalid_resampling_method(self, small_raster: Raster):
        with pytest.raises(NotImplementedError, match="Unsupported resampling method"):
            small_raster.resample(new_cell_size=2.0, method="nearest")  # pyright: ignore[reportArgumentType]

    def test_negative_cell_size_fails(self, small_raster: Raster):
        # This should fail during the internal calculations
        with pytest.raises((ValueError, RuntimeError)):
            small_raster.resample(new_cell_size=-1.0)

    def test_zero_cell_size_fails(self, small_raster: Raster):
        # This should fail during the internal calculations
        with pytest.raises((ValueError, RuntimeError, ZeroDivisionError)):
            small_raster.resample(new_cell_size=0.0)

    def test_very_small_cell_size(self, small_raster: Raster):
        # Arrange
        new_cell_size = 0.1  # Very small

        # Act
        resampled = small_raster.resample(new_cell_size)

        # Assert
        assert resampled.raster_meta.cell_size == new_cell_size
        # Should result in a very large array
        assert resampled.arr.shape[0] >= 20
        assert resampled.arr.shape[1] >= 20

    def test_metadata_preservation(self, base_raster: Raster):
        # Arrange
        original_crs = base_raster.raster_meta.crs
        new_cell_size = 5.0

        # Act
        resampled = base_raster.resample(new_cell_size)

        # Assert
        assert resampled.raster_meta.crs == original_crs
        assert resampled.raster_meta.cell_size == new_cell_size
        # Transform should be updated but maintain CRS
        assert resampled.raster_meta.transform != base_raster.raster_meta.transform

    def test_bounds_consistency(self, base_raster: Raster):
        # Arrange
        original_bounds = base_raster.bounds
        new_cell_size = 15.0

        # Act
        resampled = base_raster.resample(new_cell_size)
        new_bounds = resampled.bounds

        # Assert
        # Bounds should be similar (allowing for some discretization effects)
        # The resampled raster bounds might be slightly larger due to rounding
        tolerance = max(base_raster.raster_meta.cell_size, new_cell_size) * 2

        assert abs(new_bounds[0] - original_bounds[0]) <= tolerance  # xmin
        assert abs(new_bounds[1] - original_bounds[1]) <= tolerance  # ymin
        assert abs(new_bounds[2] - original_bounds[2]) <= tolerance  # xmax
        assert abs(new_bounds[3] - original_bounds[3]) <= tolerance  # ymax

    def test_return_type(self, small_raster: Raster):
        # Act
        result = small_raster.resample(new_cell_size=2.0)

        # Assert
        assert isinstance(result, Raster)
        assert result is not small_raster  # Should be a new instance

    def test_original_raster_unchanged(self, small_raster: Raster):
        # Arrange
        original_array = small_raster.arr.copy()
        original_cell_size = small_raster.raster_meta.cell_size

        # Act
        _ = small_raster.resample(new_cell_size=2.0)

        # Assert
        np.testing.assert_array_equal(small_raster.arr, original_array)
        assert small_raster.raster_meta.cell_size == original_cell_size

    def test_with_nan_values(self):
        # Arrange
        meta = RasterMeta(
            cell_size=10.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0),
        )
        cell_array = np.array([[1.0, np.nan], [np.nan, 4.0]])
        raster = Raster(arr=cell_array, raster_meta=meta)

        # Act
        resampled = raster.resample(new_cell_size=5.0)

        # Assert
        assert isinstance(resampled, Raster)
        assert resampled.raster_meta.cell_size == 5.0
        # Should handle NaN values gracefully
        assert not np.all(np.isnan(resampled.arr))  # Some non-NaN values

    def test_float_precision_cell_size(self, small_raster: Raster):
        # Arrange
        new_cell_size = 3.7  # Non-integer value

        # Act
        resampled = small_raster.resample(new_cell_size)

        # Assert
        assert resampled.raster_meta.cell_size == new_cell_size
        assert isinstance(resampled, Raster)

    def test_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that resample() preserves dtype."""
        result = float32_raster.resample(new_cell_size=0.5)
        assert result.arr.dtype == np.float32

    def test_preserves_dtype_float64(self, float64_raster: Raster):
        """Test that resample() preserves dtype for float64."""
        result = float64_raster.resample(new_cell_size=0.5)
        assert result.arr.dtype == np.float64


class TestExplore:
    @pytest.fixture
    def explore_map(self):
        # Hard-coded test data and simple raster with known min/max
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        raster = Raster(arr=arr, raster_meta=meta)
        return raster.explore(cbar_label="My Legend")

    def test_overlay(self, explore_map: folium.Map):
        import folium.raster_layers

        m = explore_map
        # Assert: an ImageOverlay is present
        has_image_overlay = any(
            isinstance(child, folium.raster_layers.ImageOverlay)
            for child in m._children.values()
        )
        assert has_image_overlay, "Expected an ImageOverlay to be added to the map"

    def test_cbar(self, explore_map: folium.Map):
        from branca.colormap import LinearColormap

        m = explore_map
        expected_min = 1.0
        expected_max = 4.0
        # Assert: a LinearColormap legend is present with expected properties
        legends = [
            child for child in m._children.values() if isinstance(child, LinearColormap)
        ]
        assert len(legends) >= 1, "Expected a LinearColormap legend to be added"
        legend = legends[-1]

        # Caption is set from cbar_label
        assert getattr(legend, "caption", None) == "My Legend"

        # vmin/vmax should reflect original data range (not normalized)
        assert pytest.approx(legend.vmin) == expected_min
        assert pytest.approx(legend.vmax) == expected_max

    def test_explore_without_folium_raises(self, monkeypatch: pytest.MonkeyPatch):
        # Arrange a minimal raster
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        raster = Raster(arr=arr, raster_meta=meta)

        # Simulate Folium not installed
        monkeypatch.setattr("rastr.raster.FOLIUM_INSTALLED", False, raising=False)

        # Act / Assert
        with pytest.raises(ImportError, match=r"folium.*required"):
            raster.explore()

    def test_explore_without_matplotlib_raises(self, monkeypatch: pytest.MonkeyPatch):
        # Arrange a minimal raster
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        raster = Raster(arr=arr, raster_meta=meta)

        # Simulate matplotlib not installed
        monkeypatch.setattr("rastr.raster.MATPLOTLIB_INSTALLED", False, raising=False)

        # Act / Assert
        with pytest.raises(ImportError, match=r"matplotlib.*required"):
            raster.explore()

    def test_homogenous_raster(self):
        import folium

        # Arrange a homogeneous raster
        arr = np.array([[1.0, 1.0], [1.0, 1.0]])
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        map_ = raster.explore()

        # Assert
        assert isinstance(map_, folium.Map)
        assert len(map_._children) > 0  # Check that something was added to the map

    def test_negative_x_scaling(self):
        import folium

        # Arrange a raster with negative x scaling
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        raster = Raster(arr=arr, raster_meta=meta)

        # Act
        map_ = raster.explore()

        # Assert
        assert isinstance(map_, folium.Map)
        assert len(map_._children) > 0  # Check that something was added to the map

    def test_flip_called_for_negx_scaling(self):
        # Arrange a raster that should trigger only x-flip (a < 0, e < 0)
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(-1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        raster = Raster(arr=arr, raster_meta=meta)

        # Patch np.flip as used in module under test
        with patch("rastr.raster.np.flip", wraps=np.flip) as mock_flip:
            _ = raster.explore()

        # Assert flip called exactly once (x-axis flip only)
        assert mock_flip.call_count == 1

    def test_flip_called_twice_for_negx_posy_scaling(self):
        # Arrange a raster that should trigger both x-flip and y-flip (a < 0, e > 0)
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(-1.0, 0.0, 0.0, 0.0, 1.0, 2.0),
        )
        raster = Raster(arr=arr, raster_meta=meta)

        # Patch np.flip as used in module under test
        with patch("rastr.raster.np.flip", wraps=np.flip) as mock_flip:
            _ = raster.explore()

        # Assert flip called exactly twice (both axes)
        assert mock_flip.call_count == 2

    def test_vmin_vmax_parameters(self, small_raster: Raster):
        import folium
        from branca.colormap import LinearColormap

        # Act
        map_ = small_raster.explore(vmin=2.0, vmax=3.0)

        # Assert
        assert isinstance(map_, folium.Map)
        assert len(map_._children) > 0  # Check that something was added to the map

        # Check that the legend reflects the specified vmin/vmax
        legends = [
            child
            for child in map_._children.values()
            if isinstance(child, LinearColormap)
        ]
        assert len(legends) >= 1, "Expected a LinearColormap legend to be added"
        legend = legends[-1]

        assert pytest.approx(legend.vmin) == 2.0
        assert pytest.approx(legend.vmax) == 3.0

    def test_vmin_greater_than_vmax_raises(self, small_raster: Raster):
        # Act / Assert
        with pytest.raises(ValueError, match=r"'vmin' must be less than 'vmax'"):
            small_raster.explore(vmin=3.0, vmax=2.0)


class TestRasterStatistics:
    """Test the statistical methods of the Raster class."""

    @pytest.mark.parametrize(
        ("method_name", "expected_result", "expected_result_with_nans"),
        [
            ("min", 1.0, 1.0),
            ("max", 9.0, 9.0),
            ("mean", 5.0, pytest.approx(5.2857, abs=1e-4)),
            (
                "std",
                pytest.approx(2.5820, abs=1e-4),
                pytest.approx(
                    np.nanstd(
                        np.array(
                            [[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]]
                        )
                    )
                ),
            ),
            ("median", pytest.approx(5.0), pytest.approx(6.0)),
        ],
    )
    def test_basic_statistics(
        self,
        stats_test_raster: Raster,
        stats_test_raster_with_nans: Raster,
        method_name: str,
        expected_result: float,
        expected_result_with_nans: float,
    ) -> None:
        """Test that statistical methods return the correct values and handle NaNs
        properly."""
        # Get the method from the raster object
        method = getattr(stats_test_raster, method_name)
        method_with_nans = getattr(stats_test_raster_with_nans, method_name)

        # Call the method and check results
        assert method() == expected_result
        assert method_with_nans() == expected_result_with_nans

    @pytest.mark.parametrize(
        ("quantile", "expected_result", "expected_result_with_nans"),
        [
            (0.0, pytest.approx(1.0), pytest.approx(1.0)),
            (0.5, pytest.approx(5.0), pytest.approx(6.0)),
            (1.0, pytest.approx(9.0), pytest.approx(9.0)),
        ],
    )
    def test_quantile(
        self,
        stats_test_raster: Raster,
        stats_test_raster_with_nans: Raster,
        quantile: float,
        expected_result: float,
        expected_result_with_nans: float,
    ) -> None:
        """Test the quantile method with various quantiles and NaN handling."""
        assert stats_test_raster.quantile(quantile) == expected_result
        assert stats_test_raster_with_nans.quantile(quantile) == (
            expected_result_with_nans
        )


class TestNormalize:
    def test_example(self, example_raster: Raster):
        # Act
        normalized_raster = example_raster.normalize()

        # Assert
        assert isinstance(normalized_raster, Raster)
        np.testing.assert_array_equal(normalized_raster.min(), 0.0)  # Min should be 0
        np.testing.assert_array_equal(normalized_raster.max(), 1.0)  # Max should be 1
        np.testing.assert_allclose(
            normalized_raster.arr,
            np.array([[0.0, 1 / 3], [2 / 3, 1.0]]),
        )

    def test_vmin_vmax(self, example_raster: Raster):
        # Act
        normalized_raster = example_raster.normalize(vmin=2.0, vmax=4.0)

        # Assert
        assert isinstance(normalized_raster, Raster)
        np.testing.assert_allclose(
            normalized_raster.arr,
            np.array([[0.0, 0.0], [0.5, 1.0]]),
        )

    def test_preserves_dtype_float32(self, float32_raster: Raster):
        """Test that normalize() preserves dtype."""
        result = float32_raster.normalize()
        assert result.arr.dtype == np.float32

    def test_preserves_dtype_float64(self, float64_raster: Raster):
        """Test that normalize() preserves dtype for float64."""
        result = float64_raster.normalize()
        assert result.arr.dtype == np.float64

    def test_preserves_dtype_float16(self, float16_raster: Raster):
        """Test that normalize() preserves dtype for float16."""
        result = float16_raster.normalize()
        assert result.arr.dtype == np.float16
