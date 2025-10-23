from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import rasterio
import rasterio.merge
from pyproj.crs.crs import CRS

from rastr.meta import RasterMeta
from rastr.raster import Raster

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    from rasterio._err import CPLE_BaseError
except ImportError:
    CPLE_BaseError = Exception  # Fallback if private module import fails

R = TypeVar("R", bound=Raster)


def read_raster_inmem(
    raster_path: Path | str,
    *,
    crs: CRS | str | None = None,
    cls: type[R] = Raster,
) -> R:
    """Read raster data from a file and return an in-memory Raster object.

    Args:
        raster_path: Path to the raster file.
        crs: Optional CRS to override the raster's native CRS.
        cls: The Raster subclass to instantiate. This is mostly for internal use,
             but can be useful if you have a custom `Raster` subclass.
    """
    crs = CRS.from_user_input(crs) if crs is not None else None

    with rasterio.open(raster_path, mode="r") as dst:
        # Read the entire array
        raw_arr: NDArray = dst.read()
        raw_arr = raw_arr.squeeze()

        # Extract metadata
        cell_size = dst.res[0]
        if crs is None:
            crs = CRS.from_user_input(dst.crs)
        transform = dst.transform
        nodata = dst.nodata

        # Cast integers to float16 to handle NaN values
        if np.issubdtype(raw_arr.dtype, np.integer):
            arr = raw_arr.astype(np.float16)
        else:
            arr = raw_arr

        if nodata is not None:
            arr[raw_arr == nodata] = np.nan

    raster_meta = RasterMeta(cell_size=cell_size, crs=crs, transform=transform)
    raster_obj = cls(arr=arr, raster_meta=raster_meta)
    return raster_obj


def read_raster_mosaic_inmem(
    mosaic_dir: Path | str, *, glob: str = "*.tif", crs: CRS | None = None
) -> Raster:
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
        raw_arr = arr.squeeze()

        # Cast integers to float16 to handle NaN values
        if np.issubdtype(raw_arr.dtype, np.integer):
            arr = raw_arr.astype(np.float16)
        else:
            arr = raw_arr

        if nodata is not None:
            arr[raw_arr == nodata] = np.nan

        raster_meta = RasterMeta(cell_size=cell_size, crs=crs, transform=transform)
        raster_obj = Raster(arr=arr, raster_meta=raster_meta)
        return raster_obj
    finally:
        for src in sources:
            src.close()


def write_raster(raster: Raster, *, path: Path | str, **kwargs: Any) -> None:
    """Write the raster to a file.

    Args:
        raster: The Raster object to write.
        path: Path to output file.
        **kwargs: Additional keyword arguments to pass to `rasterio.open()`. If
                    `nodata` is provided, NaN values in the raster will be replaced
                    with the nodata value.
    """
    path = Path(path)

    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        driver = "GTiff"
    elif suffix in (".grd"):
        # https://grapherhelp.goldensoftware.com/subsys/ascii_grid_file_format.htm
        # e.g. Used by AnAqSim
        driver = "GSAG"
    else:
        msg = f"Unsupported file extension: {suffix}"
        raise ValueError(msg)

    # Handle nodata: use provided value or default to np.nan
    if "nodata" in kwargs:
        # Replace NaN values with the nodata value
        nodata_value = kwargs.pop("nodata")
        arr_to_write = np.where(np.isnan(raster.arr), nodata_value, raster.arr)
    else:
        nodata_value = np.nan
        arr_to_write = raster.arr

    with rasterio.open(
        path,
        "w",
        driver=driver,
        height=raster.arr.shape[0],
        width=raster.arr.shape[1],
        count=1,
        dtype=raster.arr.dtype,
        crs=raster.raster_meta.crs,
        transform=raster.raster_meta.transform,
        nodata=nodata_value,
        **kwargs,
    ) as dst:
        try:
            dst.write(arr_to_write, 1)
        except CPLE_BaseError as err:
            msg = f"Failed to write raster to file: {err}"
            raise OSError(msg) from err
