from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from pydantic import ValidationError
from pyproj.crs.crs import CRS
from shapely.geometry import Polygon

from rastr.meta import RasterMeta
from rastr.raster import RasterModel


@pytest.fixture(scope="module")
def example_raster():
    meta = RasterMeta(
        cell_size=1.0,
        crs=CRS.from_epsg(2193),
        transform=Affine(2.0, 0.0, 0.0, 0.0, 2.0, 0.0),
    )
    arr = np.array([[1, 2], [3, 4]], dtype=float)

    return RasterModel(arr=arr, raster_meta=meta)


@pytest.fixture(scope="module")
def example_neg_scaled_raster():
    meta = RasterMeta(
        cell_size=1.0,
        crs=CRS.from_epsg(2193),
        transform=Affine(2.0, 0.0, 0.0, 0.0, -2.0, 0.0),
    )
    arr = np.array([[1, 2], [3, 4]])

    return RasterModel(arr=arr, raster_meta=meta)


@pytest.fixture(scope="module")
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


class TestSample:
    def test_sample_nan_raise(self, example_raster: RasterModel):
        with pytest.raises(ValueError, match="NaN value found in input coordinates"):
            example_raster.sample([(0, 0), (1, np.nan)], na_action="raise")

    def test_sample_nan_ignore(self, example_raster: RasterModel):
        np.testing.assert_array_equal(
            example_raster.sample([(0, 0), (2, 2), (2, np.nan)], na_action="ignore"),
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
                irrelevant_field="irrelevant",
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


class TestRasterModel:
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
                raster + "hello"

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
                raster * "hello"

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
                raster / "hello"

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

    class TestPlot:
        def test_cell_array_unchanged(self, example_raster_with_zeros: RasterModel):
            # Arrange
            original_array = example_raster_with_zeros.arr.copy()

            # Act
            example_raster_with_zeros.plot()

            # Assert
            np.testing.assert_array_equal(example_raster_with_zeros.arr, original_array)

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
                raster_with_nas = example_raster.model_copy()
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


class TestResample:
    @pytest.fixture
    def base_raster(self):
        meta = RasterMeta(
            cell_size=10.0,  # 10-meter cells
            crs=CRS.from_epsg(2193),
            transform=Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0),  # Standard NZTM-like
        )
        # Create a 4x4 raster with values 1-16
        arr = np.arange(1, 17, dtype=float).reshape(4, 4)
        return RasterModel(arr=arr, raster_meta=meta)

    @pytest.fixture
    def small_raster(self):
        meta = RasterMeta(
            cell_size=5.0,
            crs=CRS.from_epsg(2193),
            transform=Affine(5.0, 0.0, 0.0, 0.0, -5.0, 10.0),
        )
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        return RasterModel(arr=arr, raster_meta=meta)

    def test_upsampling_doubles_resolution(self, base_raster):
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

    def test_downsampling_halves_resolution(self, base_raster):
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

    def test_same_cell_size_returns_similar_raster(self, base_raster):
        # Arrange
        original_cell_size = base_raster.raster_meta.cell_size

        # Act
        resampled = base_raster.resample(original_cell_size)

        # Assert
        assert resampled.raster_meta.cell_size == original_cell_size
        # Dimensions should be the same or very close due to discretization
        assert abs(resampled.arr.shape[0] - base_raster.arr.shape[0]) <= 1
        assert abs(resampled.arr.shape[1] - base_raster.arr.shape[1]) <= 1

    def test_extreme_upsampling(self, small_raster):
        # Arrange
        new_cell_size = 1.0  # Much smaller than original 5.0

        # Act
        resampled = small_raster.resample(new_cell_size)

        # Assert
        assert resampled.raster_meta.cell_size == new_cell_size
        # Should be significantly larger
        assert resampled.arr.shape[0] >= 8
        assert resampled.arr.shape[1] >= 8

    def test_extreme_downsampling(self, base_raster):
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

    def test_transform_scaling(self, small_raster):
        # Arrange
        new_cell_size = 2.5  # Half the original cell size

        # Act
        resampled = small_raster.resample(new_cell_size)

        # Assert
        new_transform = resampled.raster_meta.transform
        # The transform scale should be updated to reflect new cell size
        assert abs(abs(new_transform.a) - new_cell_size) < 0.1
        assert abs(abs(new_transform.e) - new_cell_size) < 0.1

    def test_bilinear_interpolation_smoothing(self, small_raster):
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

    def test_invalid_resampling_method(self, small_raster):
        with pytest.raises(NotImplementedError, match="Unsupported resampling method"):
            small_raster.resample(new_cell_size=2.0, method="nearest")

    def test_negative_cell_size_fails(self, small_raster):
        # This should fail during the internal calculations
        with pytest.raises((ValueError, RuntimeError)):
            small_raster.resample(new_cell_size=-1.0)

    def test_zero_cell_size_fails(self, small_raster):
        # This should fail during the internal calculations
        with pytest.raises((ValueError, RuntimeError, ZeroDivisionError)):
            small_raster.resample(new_cell_size=0.0)

    def test_very_small_cell_size(self, small_raster):
        # Arrange
        new_cell_size = 0.1  # Very small

        # Act
        resampled = small_raster.resample(new_cell_size)

        # Assert
        assert resampled.raster_meta.cell_size == new_cell_size
        # Should result in a very large array
        assert resampled.arr.shape[0] >= 20
        assert resampled.arr.shape[1] >= 20

    def test_metadata_preservation(self, base_raster):
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

    def test_bounds_consistency(self, base_raster):
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

    def test_return_type(self, small_raster):
        # Act
        result = small_raster.resample(new_cell_size=2.0)

        # Assert
        assert isinstance(result, RasterModel)
        assert result is not small_raster  # Should be a new instance

    def test_original_raster_unchanged(self, small_raster):
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

    def test_float_precision_cell_size(self, small_raster):
        # Arrange
        new_cell_size = 3.7  # Non-integer value

        # Act
        resampled = small_raster.resample(new_cell_size)

        # Assert
        assert resampled.raster_meta.cell_size == new_cell_size
        assert isinstance(resampled, RasterModel)
