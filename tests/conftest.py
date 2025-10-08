from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from pyproj.crs.crs import CRS

from rastr.meta import RasterMeta
from rastr.raster import Raster

collect_ignore_glob = ["assets/**"]
pytest_plugins = []


@pytest.fixture(scope="session")
def assets_dir() -> Path:
    """Return a path to the test assets directory."""
    return Path(__file__).parent / "assets"


@pytest.fixture
def float32_raster() -> Raster:
    """Create a float32 raster for testing dtype preservation."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    meta = RasterMeta(
        cell_size=1.0,
        crs=CRS.from_epsg(2193),
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
    )
    return Raster(arr=arr, raster_meta=meta)


@pytest.fixture
def float64_raster() -> Raster:
    """Create a float64 raster for testing dtype preservation."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    meta = RasterMeta(
        cell_size=1.0,
        crs=CRS.from_epsg(2193),
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
    )
    return Raster(arr=arr, raster_meta=meta)


@pytest.fixture
def float16_raster() -> Raster:
    """Create a float16 raster for testing dtype preservation."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    meta = RasterMeta(
        cell_size=1.0,
        crs=CRS.from_epsg(2193),
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
    )
    return Raster(arr=arr, raster_meta=meta)
