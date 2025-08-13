from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from pyproj.crs import CRS

from rastr.meta import RasterMeta
from rastr.raster import RasterModel


def read_raster_inmem(raster_path: Path, crs: CRS | None = None) -> RasterModel:
    """Read raster data from a file and return an in-memory Raster object."""
    with rasterio.open(raster_path, mode="r") as dst:
        # Read the entire array
        arr: NDArray[np.float64] = dst.read()
        arr = arr.squeeze().astype(np.float64)
        # Extract metadata
        cell_size = dst.res[0]
        if crs is None:
            crs = CRS.from_user_input(dst.crs)
        transform = dst.transform
        nodata = dst.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan

    raster_meta = RasterMeta(cell_size=cell_size, crs=crs, transform=transform)
    raster_obj = RasterModel(arr=arr, raster_meta=raster_meta)
    return raster_obj
