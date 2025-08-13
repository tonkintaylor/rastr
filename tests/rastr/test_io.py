from pathlib import Path

import pytest
from affine import Affine
from pyproj.crs.crs import CRS

from rastr.io import read_raster_inmem

_WGS84_CRS = CRS.from_epsg(4326)


class TestIO:
    def test_read_raster(self, assets_dir: Path):
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

    def test_read_grd_file(self, assets_dir: Path):
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
