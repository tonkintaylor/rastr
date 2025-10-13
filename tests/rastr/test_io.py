from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio.transform
from affine import Affine
from pyproj.crs.crs import CRS

from rastr.io import read_raster_inmem, read_raster_mosaic_inmem

if TYPE_CHECKING:
    from pathlib import Path

_WGS84_CRS = CRS.from_epsg(4326)


class TestReadRasterInMem:
    def test_small_tif(self, assets_dir: Path):
        raster_path = assets_dir / "pga_g_clipped.tif"
        raster_obj = read_raster_inmem(raster_path)

        assert raster_obj.arr.shape == (2, 2)
        assert raster_obj.raster_meta.cell_size == 0.01495024875620743
        assert raster_obj.raster_meta.crs.to_epsg() == 4326
        assert raster_obj.raster_meta.transform == Affine(
            0.01495024875620743,
            0.0,
            173.7895771144279,
            0.0,
            -0.014950248756214535,
            -41.82587064676616,
        )

        assert raster_obj.arr[0, 0] == pytest.approx(0.39016372)
        assert raster_obj.arr[1, 1] == pytest.approx(0.4411124)
        assert raster_obj.arr[0, 1] == pytest.approx(0.44069204)
        assert raster_obj.arr[1, 0] == pytest.approx(0.41911235)

    def test_small_grd(self, assets_dir: Path):
        raster_path = assets_dir / "pga_g_clipped.grd"
        raster_obj = read_raster_inmem(raster_path)

        assert raster_obj.arr.shape == (2, 2)
        assert raster_obj.raster_meta.cell_size == 0.01495024875620743
        assert raster_obj.raster_meta.crs.to_epsg() == 4326
        assert raster_obj.raster_meta.transform == Affine(
            0.01495024875620743,
            0.0,
            173.7895771144279,
            0.0,
            -0.014950248756214535,
            -41.82587064676616,
        )

        assert raster_obj.arr[0, 0] == pytest.approx(0.39016372)
        assert raster_obj.arr[1, 1] == pytest.approx(0.4411124)
        assert raster_obj.arr[0, 1] == pytest.approx(0.44069204)
        assert raster_obj.arr[1, 0] == pytest.approx(0.41911235)

    def test_str_crs(self, assets_dir: Path):
        # Arrange
        raster_path = assets_dir / "pga_g_clipped.tif"
        crs = "EPSG:4326"

        # Act
        raster_obj = read_raster_inmem(raster_path, crs=crs)

        # Assert
        assert raster_obj.raster_meta.crs.to_epsg() == 4326

    def test_pyproj_crs(self, assets_dir: Path):
        # Arrange
        raster_path = assets_dir / "pga_g_clipped.tif"
        crs = CRS.from_epsg(4326)

        # Act
        raster_obj = read_raster_inmem(raster_path, crs=crs)

        # Assert
        assert raster_obj.raster_meta.crs.to_epsg() == 4326

    def test_dtype_preservation_float32(self, tmp_path: Path):
        """Test that float32 dtype is preserved when reading."""
        # Arrange
        arr = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        transform = rasterio.transform.Affine(1, 0, 0, 0, -1, 2)
        path = TestReadRasterMosaicInMem._write_tiff(
            tmp_path, "test.tif", arr, transform
        )

        # Act
        raster = read_raster_inmem(path)

        # Assert
        assert raster.arr.dtype == np.float32
        np.testing.assert_allclose(raster.arr, arr)

    def test_dtype_preservation_float64(self, tmp_path: Path):
        """Test that float64 dtype is preserved when reading."""
        # Arrange
        arr = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
        transform = rasterio.transform.Affine(1, 0, 0, 0, -1, 2)
        path = TestReadRasterMosaicInMem._write_tiff(
            tmp_path, "test.tif", arr, transform
        )

        # Act
        raster = read_raster_inmem(path)

        # Assert
        assert raster.arr.dtype == np.float64
        np.testing.assert_allclose(raster.arr, arr)

    def test_integer_conversion_to_float16(self, tmp_path: Path):
        """Test integer dtypes are converted to float16 and nodata values become NaN."""
        # Arrange
        arr = np.array([[1, 2], [3, -9999]], dtype=np.int32)
        transform = rasterio.transform.Affine(1, 0, 0, 0, -1, 2)
        path = TestReadRasterMosaicInMem._write_tiff(
            tmp_path, "test.tif", arr, transform, nodata=-9999
        )

        # Act
        raster = read_raster_inmem(path)

        # Assert
        assert raster.arr.dtype == np.float16
        assert np.isnan(raster.arr[1, 1])
        expected = np.array([[1, 2], [3, np.nan]], dtype=np.float16)
        np.testing.assert_allclose(raster.arr, expected, equal_nan=True)


class TestReadRasterMosaicInMem:
    @staticmethod
    def _write_tiff(
        dir_path: Path,
        name: str,
        arr: np.ndarray,
        transform: rasterio.transform.Affine,
        crs: str | CRS = "EPSG:4326",
        nodata: float | None = float("nan"),
    ) -> Path:
        path = dir_path / name
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=arr.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(arr, 1)
        return path

    def test_basic_and_nodata(self, tmp_path: Path):
        # Arrange
        tile1 = np.array([[1, 2], [3, -9999]], dtype=np.float32)
        tile2 = np.array([[5, 6], [7, 8]], dtype=np.float32)
        transform1 = rasterio.transform.Affine(1, 0, 0, 0, -1, 2)
        transform2 = rasterio.transform.Affine(1, 0, 2, 0, -1, 2)
        self._write_tiff(tmp_path, "tile1.tif", tile1, transform1, nodata=-9999)
        self._write_tiff(tmp_path, "tile2.tif", tile2, transform2)

        # Act
        raster = read_raster_mosaic_inmem(tmp_path)

        # Assert
        assert raster.arr.shape == (2, 4)
        assert raster.raster_meta.transform == transform1
        assert raster.raster_meta.cell_size == 1
        assert np.isnan(raster.arr[1, 1])
        expected = np.array([[1, 2, 5, 6], [3, np.nan, 7, 8]], dtype=np.float32)
        np.testing.assert_allclose(raster.arr, expected, equal_nan=True)

    def test_custom_glob(self, tmp_path: Path):
        # Arrange
        arr_a = np.array([[10, 11], [12, 13]], dtype=np.float32)
        arr_b = np.array([[20, 21], [22, 23]], dtype=np.float32)
        transform_a = rasterio.transform.Affine(1, 0, 0, 0, -1, 2)
        transform_b = rasterio.transform.Affine(1, 0, 2, 0, -1, 2)
        self._write_tiff(tmp_path, "keep_a.tif", arr_a, transform_a)
        self._write_tiff(tmp_path, "skip_b.tif", arr_b, transform_b)

        # Act
        raster = read_raster_mosaic_inmem(tmp_path, glob="keep_*.tif")

        # Assert
        assert raster.arr.shape == (2, 2)
        np.testing.assert_allclose(raster.arr, arr_a)

    def test_no_files(self, tmp_path: Path):
        # Arrange / Act / Assert
        with pytest.raises(FileNotFoundError):
            read_raster_mosaic_inmem(tmp_path)

    def test_override_crs(self, tmp_path: Path):
        # Arrange
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        transform = rasterio.transform.Affine(1, 0, 0, 0, -1, 2)
        self._write_tiff(tmp_path, "tile.tif", arr, transform)
        override_crs = CRS.from_epsg(3857)

        # Act
        raster = read_raster_mosaic_inmem(tmp_path, crs=override_crs)

        # Assert
        assert raster.raster_meta.crs.to_epsg() == 3857
        np.testing.assert_allclose(raster.arr, arr)

    def test_vertical_adjacency(self, tmp_path: Path):
        # Arrange
        top = np.array([[1, 2], [3, 4]], dtype=np.float32)
        bottom = np.array([[5, 6], [7, 8]], dtype=np.float32)
        transform_top = rasterio.transform.Affine(1, 0, 0, 0, -1, 4)
        transform_bottom = rasterio.transform.Affine(1, 0, 0, 0, -1, 2)
        self._write_tiff(tmp_path, "top.tif", top, transform_top)
        self._write_tiff(tmp_path, "bottom.tif", bottom, transform_bottom)

        # Act
        raster = read_raster_mosaic_inmem(tmp_path)

        # Assert
        assert raster.arr.shape == (4, 2)
        assert raster.raster_meta.transform == transform_top
        expected = np.vstack([top, bottom])
        np.testing.assert_allclose(raster.arr, expected)

    def test_overlapping_first_wins(self, tmp_path: Path):
        # Arrange
        first = np.array([[1, 1], [1, 1]], dtype=np.float32)
        second = np.array([[2, 2], [2, 2]], dtype=np.float32)
        transform = rasterio.transform.Affine(1, 0, 0, 0, -1, 2)
        # File naming ensures "a_" is encountered before "z_" in glob listing
        self._write_tiff(tmp_path, "a_first.tif", first, transform)
        self._write_tiff(tmp_path, "z_second.tif", second, transform)

        # Act
        raster = read_raster_mosaic_inmem(tmp_path)

        # Assert
        np.testing.assert_allclose(raster.arr, first)

    def test_mosaic_dtype_preservation_float32(self, tmp_path: Path):
        """Test that float32 dtype is preserved when reading mosaics."""
        # Arrange
        tile1 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        tile2 = np.array([[5.5, 6.5], [7.5, 8.5]], dtype=np.float32)
        transform1 = rasterio.transform.Affine(1, 0, 0, 0, -1, 2)
        transform2 = rasterio.transform.Affine(1, 0, 2, 0, -1, 2)
        self._write_tiff(tmp_path, "tile1.tif", tile1, transform1)
        self._write_tiff(tmp_path, "tile2.tif", tile2, transform2)

        # Act
        raster = read_raster_mosaic_inmem(tmp_path)

        # Assert
        assert raster.arr.dtype == np.float32

    def test_mosaic_integer_conversion_to_float16(self, tmp_path: Path):
        """Test that integer dtypes are converted to float16 in mosaics."""
        # Arrange
        tile1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        tile2 = np.array([[5, 6], [7, 8]], dtype=np.int32)
        transform1 = rasterio.transform.Affine(1, 0, 0, 0, -1, 2)
        transform2 = rasterio.transform.Affine(1, 0, 2, 0, -1, 2)
        self._write_tiff(tmp_path, "tile1.tif", tile1, transform1, nodata=None)
        self._write_tiff(tmp_path, "tile2.tif", tile2, transform2, nodata=None)

        # Act
        raster = read_raster_mosaic_inmem(tmp_path)

        # Assert
        assert raster.arr.dtype == np.float16
