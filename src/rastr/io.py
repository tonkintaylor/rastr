from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
import rasterio.merge
from pyproj.crs.crs import CRS

from rastr.meta import RasterMeta
from rastr.raster import RasterModel

if TYPE_CHECKING:
    from numpy.typing import NDArray


def read_raster_inmem(
    raster_path: Path | str, *, crs: CRS | str | None = None
) -> RasterModel:
    """Read raster data from a file and return an in-memory Raster object."""
    crs = CRS.from_user_input(crs) if crs is not None else None

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


def read_raster_mosaic_inmem(
    mosaic_dir: Path | str, *, glob: str = "*.tif", crs: CRS | None = None
) -> RasterModel:
    """Read a raster mosaic from a directory and return an in-memory Raster object.

    This assumes that all rasters have the same metadata, e.g. coordinate system,
    cell size, etc.
    """
    mosaic_dir = Path(mosaic_dir)
    raster_paths = list(mosaic_dir.glob(glob))
    if not raster_paths:
        msg = f"No raster files found in {mosaic_dir} matching {glob}"
        raise FileNotFoundError(msg)

    # Sort raster_paths in alphabetical order by stem
    raster_paths.sort(key=lambda p: p.stem)

    # Open all TIFF datasets using context managers to ensure proper closure
    sources = []
    try:
        for raster_path in raster_paths:
            src = rasterio.open(raster_path)
            sources.append(src)

        # Merge into a single mosaic array & transform
        arr, transform = rasterio.merge.merge(sources)

        # Copy metadata from the first dataset
        out_meta = sources[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": arr.shape[1],
                "width": arr.shape[2],
                "transform": transform,
            }
        )
        cell_size = sources[0].res[0]
        if crs is None:
            crs = CRS.from_user_input(sources[0].crs)

        nodata = sources[0].nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan

        arr = arr.squeeze().astype(np.float64)

        raster_meta = RasterMeta(cell_size=cell_size, crs=crs, transform=transform)
        raster_obj = RasterModel(arr=arr, raster_meta=raster_meta)
        return raster_obj
    finally:
        for src in sources:
            src.close()
