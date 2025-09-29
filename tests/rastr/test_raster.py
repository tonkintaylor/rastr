from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from unittest.mock import patch

import folium
import folium.raster_layers
import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from branca.colormap import LinearColormap
from pydantic import ValidationError
from pyproj.crs.crs import CRS
from shapely import MultiPolygon, box
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from rastr.meta import RasterMeta
from rastr.raster import RasterModel

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def example_raster():
    meta = RasterMeta(
        cell_size=1.0,
        crs=CRS.from_epsg(2193),
        transform=Affine(2.0, 0.0, 0.0, 0.0, 2.0, 0.0),
    )
    arr = np.array([[1, 2], [3, 4]], dtype=float)

    return RasterModel(arr=arr, raster_meta=meta)


@pytest.fixture
def example_neg_scaled_raster():
    meta = RasterMeta(
        cell_size=1.0,
        crs=CRS.from_epsg(2193),
        transform=Affine(2.0, 0.0, 0.0, 0.0, -2.0, 0.0),
    )
    arr = np.array([[1, 2], [3, 4]])

    return RasterModel(arr=arr, raster_meta=meta)


@pytest.fixture
def example_raster_with_zeros():
    meta = RasterMeta(
        cell_size=1.0,
        crs=CRS.from_epsg(2193),
        transform=Affine(2.0, 0.0, 0.0, 0.0, 2.0, 0.0),
    )
    arr = np.array([[1, 0], [0, 4]], dtype=float)

    return RasterModel(
        arr=arr,
        raster_meta=meta,
    )


class TestRasterModel:
    class TestInit:
        def test_meta_and_arr(self, example_raster: RasterModel):
            # Act, Assert
            RasterModel(
                arr=example_raster.arr,
                meta=example_raster.raster_meta,
            )

        def test_both_meta_and_raster_meta(self, example_raster: RasterModel):
            # Act, Assert
            with pytest.raises(
                ValueError,
                match="Only one of 'meta' or 'raster_meta' should be provided",
            ):
                RasterModel(
                    arr=example_raster.arr,
                    meta=example_raster.raster_meta,
                    raster_meta=example_raster.raster_meta,
                )

        def test_missing_meta(self, example_raster: RasterModel):
            # Act, Assert
            with pytest.raises(
                ValueError, match="The attribute 'raster_meta' is required."
            ):
                RasterModel(arr=example_raster.arr)

    class TestMetaAlias:
        def test_meta_getter(self, example_raster: RasterModel):
            # Act
            meta_via_alias = example_raster.meta
            meta_direct = example_raster.raster_meta

            # Assert
            assert meta_via_alias is meta_direct
            assert meta_via_alias == meta_direct

        def test_meta_setter(self, example_raster: RasterModel):
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

    class TestShape:
        def test_shape_property(self, example_raster: RasterModel):
            # Act
            shape = example_raster.shape

            # Assert
            assert shape == (2, 2)
            assert shape == example_raster.arr.shape

    class TestCRS:
        def test_crs_getter(self, example_raster: RasterModel):
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

        def test_crs_setter(self, example_raster: RasterModel):
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

    class TestSample:
        def test_sample_nan_raise(self, example_raster: RasterModel):
            with pytest.raises(
                ValueError, match="NaN value found in input coordinates"
            ):
                example_raster.sample([(0, 0), (1, np.nan)], na_action="raise")

        def test_sample_nan_ignore(self, example_raster: RasterModel):
            np.testing.assert_array_equal(
                example_raster.sample(
                    [(0, 0), (2, 2), (2, np.nan)], na_action="ignore"
                ),
                [1.0, 4, np.nan],
            )

        def test_oob_query(self, example_raster: RasterModel):
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
            raster = RasterModel(
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

        def test_ndarray_input(self, example_raster: RasterModel):
            # Arrange
            coords = np.array([[0, 0], [1, 1]])

            # Act
            result = example_raster.sample(coords, na_action="raise")

            # Assert
            np.testing.assert_array_equal(result, np.array([1.0, 1.0]))

        def test_shapely_points_input(self, example_raster: RasterModel):
            # Arrange
            points = [Point(0, 0), Point(2, 2)]

            # Act
            result = example_raster.sample(points, na_action="raise")

            # Assert
            np.testing.assert_array_equal(result, np.array([1.0, 4.0]))

        def test_single_shapely_point_input(self, example_raster: RasterModel):
            # Arrange
            point = Point(0, 0)

            # Act
            result = example_raster.sample(point, na_action="raise")

            # Assert
            np.testing.assert_array_equal(result, np.array(1.0), strict=True)

        def test_single_tuple_input(self, example_raster: RasterModel):
            # Arrange
            coord = (0, 0)

            # Act
            result = example_raster.sample(coord, na_action="raise")

            # Assert
            np.testing.assert_array_equal(result, np.array(1.0), strict=True)

    class TestBounds:
        def test_bounds(self, example_raster: RasterModel):
            assert example_raster.bounds == (0.0, 0.0, 4.0, 4.0)

        def test_bounds_neg_scaled(self, example_neg_scaled_raster: RasterModel):
            assert example_neg_scaled_raster.bounds == (0.0, -4.0, 4.0, 0.0)

    class TestAsGeoDataFrame:
        def test_as_geodataframe(self, example_raster: RasterModel):
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
            raster1 = RasterModel(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            raster2 = RasterModel(
                arr=np.array([[5, 6], [7, 8]]),
                raster_meta=raster_meta,
            )

            # Act
            result = raster1 + raster2

            # Assert
            np.testing.assert_array_equal(result.arr, np.array([[6, 8], [10, 12]]))

        def test_add_subclass_return_type(self):
            # Arrange
            class MyRaster(RasterModel):
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
            raster1 = RasterModel(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta1,
            )

            raster_meta2 = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(4326),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster2 = RasterModel(
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
            raster1 = RasterModel(
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
            raster1 = RasterModel(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            raster2 = RasterModel(
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
            raster = RasterModel(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            # Act
            with pytest.raises(TypeError, match="unsupported operand type"):
                raster + "hello"  # type: ignore[reportOperatorIssue]

    class TestMul:
        def test_basic(self):
            # Arrange
            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = RasterModel(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            raster2 = RasterModel(
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
            raster1 = RasterModel(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta1,
            )

            raster_meta2 = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(4326),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster2 = RasterModel(
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
            raster1 = RasterModel(
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
            raster1 = RasterModel(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            raster2 = RasterModel(
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
            raster = RasterModel(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            # Act
            with pytest.raises(TypeError):
                raster * "hello"  # type: ignore[reportOperatorIssue]

    class TestTrueDiv:
        def test_basic(self):
            # Arrange
            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = RasterModel(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            raster2 = RasterModel(
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
            raster1 = RasterModel(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta1,
            )

            raster_meta2 = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(4326),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster2 = RasterModel(
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
            raster1 = RasterModel(
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
            raster1 = RasterModel(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            raster2 = RasterModel(
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
            raster = RasterModel(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            # Act
            with pytest.raises(TypeError):
                raster / "hello"  # type: ignore[reportOperatorIssue]

    class TestSub:
        def test_basic(self):
            # Arrange
            raster_meta = RasterMeta(
                cell_size=1.0,
                crs=CRS.from_epsg(2193),
                transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
            raster1 = RasterModel(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            raster2 = RasterModel(
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
            raster1 = RasterModel(
                arr=np.array([[1, 2], [3, 4]]),
                raster_meta=raster_meta,
            )

            # Act
            result = 1.0 - raster1

            # Assert
            np.testing.assert_array_equal(result.arr, np.array([[0, -1], [-2, -3]]))

    class TestApply:
        def test_sine(self, example_raster: RasterModel):
            # Act
            result = example_raster.apply(np.sin)

            # Assert
            np.testing.assert_array_equal(result.arr, np.sin(example_raster.arr))

    class TestToFile:
        def test_saving_gtiff(self, tmp_path: Path, example_raster: RasterModel):
            # Arrange
            filename = tmp_path / "test_raster.tif"

            # Act
            example_raster.to_file(filename)

            # Assert
            assert filename.exists()

        def test_saving_grd_file(self, tmp_path: Path, example_raster: RasterModel):
            # Arrange
            filename = tmp_path / "test_raster.grd"

            # Act
            example_raster.to_file(filename)

            # Assert
            assert filename.exists()

        def test_string_as_path(self, tmp_path: Path, example_raster: RasterModel):
            # Arrange
            filename = tmp_path / "test_raster.tif"

            # Act
            example_raster.to_file(filename.as_posix())

            # Assert
            assert filename.exists()

    class TestPlot:
        def test_cell_array_unchanged(self, example_raster_with_zeros: RasterModel):
            # Arrange
            original_array = example_raster_with_zeros.arr.copy()

            # Act
            example_raster_with_zeros.plot()

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
            raster = RasterModel(arr=arr, raster_meta=meta)

            # Simulate matplotlib not installed
            monkeypatch.setattr(
                "rastr.raster.MATPLOTLIB_INSTALLED", False, raising=False
            )

            # Act / Assert
            with pytest.raises(ImportError, match="matplotlib.*required"):
                raster.plot()

    class TestExample:
        def test_example(self):
            # Act
            raster = RasterModel.example()

            # Assert
            assert isinstance(raster, RasterModel)

    class TestFillNA:
        def test_2by2_example(self):
            # Arrange
            raster = RasterModel(
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

    class TestGetXY:
        def test_get_xy(self, example_raster: RasterModel):
            # Act
            x, y = example_raster.get_xy()

            # Assert
            # N.B. the xy coordinates in meshgrid style - 2D arrays
            expected_x = np.array([[1.0, 3.0], [1.0, 3.0]])
            expected_y = np.array([[1.0, 1.0], [3.0, 3.0]])
            np.testing.assert_array_equal(x, expected_x)
            np.testing.assert_array_equal(y, expected_y)

    class TestBlur:
        def test_numeric_propertoes(self, example_raster: RasterModel):
            # Act
            blurred_raster = example_raster.blur(sigma=1.0)

            # Assert
            assert isinstance(blurred_raster, RasterModel)

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

    class TestExtrapolate:
        class TestNearest:
            def test_no_nas_stays_the_same(self, example_raster: RasterModel):
                # Act
                extrapolated_raster = example_raster.extrapolate(method="nearest")

                # Assert
                assert isinstance(extrapolated_raster, RasterModel)
                np.testing.assert_array_equal(
                    extrapolated_raster.arr, example_raster.arr
                )

            def test_fillna(self, example_raster: RasterModel):
                # Arrange
                raster_with_nas = example_raster
                raster_with_nas.arr[0, 0] = np.nan

                # Act
                extrapolated_raster = raster_with_nas.extrapolate(method="nearest")

                # Assert
                assert isinstance(extrapolated_raster, RasterModel)
                np.testing.assert_array_equal(
                    extrapolated_raster.arr,
                    np.array(
                        [[2, 2], [3, 4]]
                    ),  # NaN should be filled with nearest value
                )

            def test_start_with_all_na(self):
                # Arrange
                raster = RasterModel(
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
                assert isinstance(extrapolated_raster, RasterModel)
                np.testing.assert_array_equal(
                    extrapolated_raster.arr,
                    np.array(
                        [[np.nan, np.nan], [np.nan, np.nan]]
                    ),  # No change expected
                )

    class TestContour:
        def test_contour_with_list_levels(self):
            # Arrange
            raster = RasterModel.example()
            levels = [0.0, 0.5]

            # Act
            contour_gdf = raster.contour(levels=levels)

            # Assert
            assert isinstance(contour_gdf, gpd.GeoDataFrame)
            assert "level" in contour_gdf.columns
            assert len(contour_gdf) >= 0  # Should return some contours or empty GDF

        def test_contour_with_ndarray_levels(self):
            # Arrange
            raster = RasterModel.example()
            levels = np.array([0.0, 0.5])

            # Act
            contour_gdf = raster.contour(levels=levels)

            # Assert
            assert isinstance(contour_gdf, gpd.GeoDataFrame)
            assert "level" in contour_gdf.columns
            assert len(contour_gdf) >= 0  # Should return some contours or empty GDF

        def test_contour_list_and_ndarray_equivalent(self):
            # Arrange
            raster = RasterModel.example()
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
            raster = RasterModel.example()
            levels = [0.0, 0.5]

            # Act - should pass without error when using positional levels arg
            contour_gdf = raster.contour(levels)  # noqa: F841

        def test_contour_returns_gdf_with_correct_columns(self):
            raster = RasterModel.example()
            gdf = raster.contour(levels=[0.0, 0.5])

            assert isinstance(gdf, gpd.GeoDataFrame)
            assert list(gdf.columns) == ["level", "geometry"]
            assert "level" in gdf.columns
            assert "geometry" in gdf.columns

        def test_contour_levels_in_result(self):
            raster = RasterModel.example()
            levels = [0.0, 0.5]
            gdf = raster.contour(levels=levels)

            result_levels = set(gdf["level"].unique())
            expected_levels = set(levels)
            assert result_levels == expected_levels

        def test_contour_dissolve_behavior_one_row_per_level(self):
            raster = RasterModel.example()
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
            raster = RasterModel.example()
            gdf = raster.contour(levels=[0.0], smoothing=True)

            assert len(gdf) > 0
            assert all(gdf["level"] == 0.0)

        def test_contour_without_smoothing(self):
            raster = RasterModel.example()
            gdf = raster.contour(levels=[0.0], smoothing=False)

            assert len(gdf) > 0
            assert all(gdf["level"] == 0.0)

        def test_level_at_max(self):
            # https://github.com/tonkintaylor/rastr/issues/154

            # Arrange
            raster = RasterModel(
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
            raster = RasterModel(
                arr=np.array([[1, 4, 4, 2], [1, 2, 4, 2], [1, 2, 4, 2], [1, 2, 4, 2]]),
                meta=RasterMeta.example(),
            )

            # Act
            gdf = raster.contour(levels=[1])

            # Assert
            assert len(gdf) > 0
            assert set(gdf["level"]) == {1.0}


@pytest.fixture
def base_raster():
    meta = RasterMeta(
        cell_size=10.0,  # 10-meter cells
        crs=CRS.from_epsg(2193),
        transform=Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0),  # Standard NZTM-like
    )
    # Create a 4x4 raster with values 1-16
    arr = np.arange(1, 17, dtype=float).reshape(4, 4)
    return RasterModel(arr=arr, raster_meta=meta)


@pytest.fixture
def small_raster():
    meta = RasterMeta(
        cell_size=5.0,
        crs=CRS.from_epsg(2193),
        transform=Affine(5.0, 0.0, 0.0, 0.0, -5.0, 10.0),
    )
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    return RasterModel(arr=arr, raster_meta=meta)


class TestCrop:
    def test_fully_within_bbox_base(self, base_raster: RasterModel):
        # Arrange
        bounds = base_raster.bounds

        # Act
        cropped = base_raster.crop(bounds)

        # Assert
        assert cropped == base_raster

    def test_fully_within_bbox_small(self, small_raster: RasterModel):
        # Arrange
        bounds = small_raster.bounds

        # Act
        cropped = small_raster.crop(bounds)

        # Assert
        assert cropped == small_raster

    def test_crop_y_only(self, base_raster: RasterModel):
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

    def test_crop_x_only(self, base_raster: RasterModel):
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

    def test_underflow_crops_border_cells(self, base_raster: RasterModel):
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

    def test_overflow_doesnt_crop(self, base_raster: RasterModel):
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
        self, base_raster: RasterModel, strategy: Literal["overflow", "underflow"]
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

    def test_overflow_crops(self, base_raster: RasterModel):
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
        self, base_raster: RasterModel, bounds: tuple[float, float, float, float]
    ):
        # Arrange, Act & Assert
        with pytest.raises(
            ValueError,
            match="Cropped array is empty; no cells within the specified bounds.",
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
        raster = RasterModel(arr=arr, raster_meta=meta)

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

    def test_unsupported_crop_strategy(self, base_raster: RasterModel):
        # Arrange
        bounds = base_raster.bounds

        # Act & Assert
        with pytest.raises(
            NotImplementedError,
            match="Unsupported cropping strategy: invalid_strategy",
        ):
            base_raster.crop(bounds, strategy="invalid_strategy")  # type: ignore[reportArgumentType]

    def test_strategy_is_keyword_only(self, base_raster: RasterModel):
        # Arrange
        bounds = base_raster.bounds

        # Act & Assert
        with pytest.raises(TypeError):
            base_raster.crop(bounds, "overflow")  # type: ignore[reportCallIssue]


class TestClip:
    def test_example(self):
        # Arrange
        raster = RasterModel(
            arr=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            raster_meta=RasterMeta.example(),
        )
        polygon = raster.bbox.buffer(-2.5)

        # Act
        clipped = raster.clip(polygon)

        # Assert
        assert isinstance(clipped, RasterModel)
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

    def test_own_bbox(self, base_raster: RasterModel):
        # Arrange
        polygon = base_raster.bbox

        # Act
        clipped = base_raster.clip(polygon)

        # Assert
        assert clipped == base_raster

    def test_multipolygon(self, base_raster: RasterModel):
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


class TestResample:
    def test_upsampling_doubles_resolution(self, base_raster: RasterModel):
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

    def test_downsampling_halves_resolution(self, base_raster: RasterModel):
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

    def test_same_cell_size_returns_similar_raster(self, base_raster: RasterModel):
        # Arrange
        original_cell_size = base_raster.raster_meta.cell_size

        # Act
        resampled = base_raster.resample(original_cell_size)

        # Assert
        assert resampled.raster_meta.cell_size == original_cell_size
        # Dimensions should be the same or very close due to discretization
        assert abs(resampled.arr.shape[0] - base_raster.arr.shape[0]) <= 1
        assert abs(resampled.arr.shape[1] - base_raster.arr.shape[1]) <= 1

    def test_extreme_upsampling(self, small_raster: RasterModel):
        # Arrange
        new_cell_size = 1.0  # Much smaller than original 5.0

        # Act
        resampled = small_raster.resample(new_cell_size)

        # Assert
        assert resampled.raster_meta.cell_size == new_cell_size
        # Should be significantly larger
        assert resampled.arr.shape[0] >= 8
        assert resampled.arr.shape[1] >= 8

    def test_extreme_downsampling(self, base_raster: RasterModel):
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

    def test_transform_scaling(self, small_raster: RasterModel):
        # Arrange
        new_cell_size = 2.5  # Half the original cell size

        # Act
        resampled = small_raster.resample(new_cell_size)

        # Assert
        new_transform = resampled.raster_meta.transform
        # The transform scale should be updated to reflect new cell size
        assert abs(abs(new_transform.a) - new_cell_size) < 0.1
        assert abs(abs(new_transform.e) - new_cell_size) < 0.1

    def test_bilinear_interpolation_smoothing(self, small_raster: RasterModel):
        # Arrange
        new_cell_size = 2.0  # Between original cells

        # Act
        resampled = small_raster.resample(new_cell_size)

        # Assert
        # With bilinear interpolation, we shouldn't have any extreme values
        # that are outside the range of the original data
        original_min = np.min(small_raster.arr)
        original_max = np.max(small_raster.arr)
        resampled_min = np.nanmin(resampled.arr)
        resampled_max = np.nanmax(resampled.arr)

        # Values should generally be within the original range
        # (allowing small numerical tolerances)
        assert resampled_min >= original_min - 0.1
        assert resampled_max <= original_max + 0.1

    def test_invalid_resampling_method(self, small_raster: RasterModel):
        with pytest.raises(NotImplementedError, match="Unsupported resampling method"):
            small_raster.resample(new_cell_size=2.0, method="nearest")  # pyright: ignore[reportArgumentType]

    def test_negative_cell_size_fails(self, small_raster: RasterModel):
        # This should fail during the internal calculations
        with pytest.raises((ValueError, RuntimeError)):
            small_raster.resample(new_cell_size=-1.0)

    def test_zero_cell_size_fails(self, small_raster: RasterModel):
        # This should fail during the internal calculations
        with pytest.raises((ValueError, RuntimeError, ZeroDivisionError)):
            small_raster.resample(new_cell_size=0.0)

    def test_very_small_cell_size(self, small_raster: RasterModel):
        # Arrange
        new_cell_size = 0.1  # Very small

        # Act
        resampled = small_raster.resample(new_cell_size)

        # Assert
        assert resampled.raster_meta.cell_size == new_cell_size
        # Should result in a very large array
        assert resampled.arr.shape[0] >= 20
        assert resampled.arr.shape[1] >= 20

    def test_metadata_preservation(self, base_raster: RasterModel):
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

    def test_bounds_consistency(self, base_raster: RasterModel):
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

    def test_return_type(self, small_raster: RasterModel):
        # Act
        result = small_raster.resample(new_cell_size=2.0)

        # Assert
        assert isinstance(result, RasterModel)
        assert result is not small_raster  # Should be a new instance

    def test_original_raster_unchanged(self, small_raster: RasterModel):
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
        raster = RasterModel(arr=cell_array, raster_meta=meta)

        # Act
        resampled = raster.resample(new_cell_size=5.0)

        # Assert
        assert isinstance(resampled, RasterModel)
        assert resampled.raster_meta.cell_size == 5.0
        # Should handle NaN values gracefully
        assert not np.all(np.isnan(resampled.arr))  # Some non-NaN values

    def test_float_precision_cell_size(self, small_raster: RasterModel):
        # Arrange
        new_cell_size = 3.7  # Non-integer value

        # Act
        resampled = small_raster.resample(new_cell_size)

        # Assert
        assert resampled.raster_meta.cell_size == new_cell_size
        assert isinstance(resampled, RasterModel)


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
        raster = RasterModel(arr=arr, raster_meta=meta)
        return raster.explore(cbar_label="My Legend")

    def test_overlay(self, explore_map: folium.Map):
        m = explore_map
        # Assert: an ImageOverlay is present
        has_image_overlay = any(
            isinstance(child, folium.raster_layers.ImageOverlay)
            for child in m._children.values()
        )
        assert has_image_overlay, "Expected an ImageOverlay to be added to the map"

    def test_cbar(self, explore_map: folium.Map):
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
        raster = RasterModel(arr=arr, raster_meta=meta)

        # Simulate Folium not installed
        monkeypatch.setattr("rastr.raster.FOLIUM_INSTALLED", False, raising=False)

        # Act / Assert
        with pytest.raises(ImportError, match="folium.*required"):
            raster.explore()

    def test_explore_without_matplotlib_raises(self, monkeypatch: pytest.MonkeyPatch):
        # Arrange a minimal raster
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        raster = RasterModel(arr=arr, raster_meta=meta)

        # Simulate matplotlib not installed
        monkeypatch.setattr("rastr.raster.MATPLOTLIB_INSTALLED", False, raising=False)

        # Act / Assert
        with pytest.raises(ImportError, match="matplotlib.*required"):
            raster.explore()

    def test_homogenous_raster(self):
        # Arrange a homogeneous raster
        arr = np.array([[1.0, 1.0], [1.0, 1.0]])
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        raster = RasterModel(arr=arr, raster_meta=meta)

        # Act
        map_ = raster.explore()

        # Assert
        assert isinstance(map_, folium.Map)
        assert len(map_._children) > 0  # Check that something was added to the map

    def test_negative_x_scaling(self):
        # Arrange a raster with negative x scaling
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        meta = RasterMeta(
            cell_size=1.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        raster = RasterModel(arr=arr, raster_meta=meta)

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
        raster = RasterModel(arr=arr, raster_meta=meta)

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
        raster = RasterModel(arr=arr, raster_meta=meta)

        # Patch np.flip as used in module under test
        with patch("rastr.raster.np.flip", wraps=np.flip) as mock_flip:
            _ = raster.explore()

        # Assert flip called exactly twice (both axes)
        assert mock_flip.call_count == 2
