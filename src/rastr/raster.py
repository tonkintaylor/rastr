"""Raster data structure."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
import rasterio.plot
import rasterio.sample
import rasterio.transform
import skimage.measure
import xyzservices.providers as xyz
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray
from pydantic import BaseModel, InstanceOf, field_validator
from pyproj.crs.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import BufferedDatasetWriter, DatasetReader, DatasetWriter, MemoryFile
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString, Polygon
from typing_extensions import Self

from rastr.arr.fill import fillna_nearest_neighbours
from rastr.gis.fishnet import create_fishnet
from rastr.gis.smooth import catmull_rom_smooth
from rastr.meta import RasterMeta

if TYPE_CHECKING:
    from folium import Map

try:
    import folium
    import folium.raster_layers
    from folium import Map
except ImportError:
    FOLIUM_INSTALLED = False
else:
    FOLIUM_INSTALLED = True

try:
    from rasterio._err import CPLE_BaseError
except ImportError:
    CPLE_BaseError = Exception  # Fallback if private module import fails


CTX_BASEMAP_SOURCE = xyz.Esri.WorldImagery  # pyright: ignore[reportAttributeAccessIssue]


class RasterCellArrayShapeError(ValueError):
    """Custom error for invalid raster cell array shapes."""


class RasterModel(BaseModel):
    """2-dimensional raster and metadata."""

    arr: InstanceOf[np.ndarray]
    raster_meta: RasterMeta

    def __eq__(self, other: object) -> bool:
        """Check equality of two RasterModel objects."""
        if not isinstance(other, RasterModel):
            return NotImplemented
        return (
            np.array_equal(self.arr, other.arr)
            and self.raster_meta == other.raster_meta
        )

    __hash__ = BaseModel.__hash__

    def __add__(self, other: float | Self) -> Self:
        if isinstance(other, float | int):
            new_arr = self.arr + other
            return RasterModel(arr=new_arr, raster_meta=self.raster_meta)
        elif isinstance(other, RasterModel):
            if self.raster_meta != other.raster_meta:
                msg = (
                    "Rasters must have the same metadata (e.g. CRS, cell size, etc.) "
                    "to be added"
                )
                raise ValueError(msg)
            if self.arr.shape != other.arr.shape:
                msg = (
                    "Rasters must have the same shape to be added:\n"
                    f"{self.arr.shape} != {other.arr.shape}"
                )
                raise ValueError(msg)
            new_arr = self.arr + other.arr
            return RasterModel(arr=new_arr, raster_meta=self.raster_meta)
        else:
            return NotImplemented

    def __radd__(self, other: float) -> Self:
        return self + other

    def __mul__(self, other: float | Self) -> Self:
        if isinstance(other, float | int):
            new_arr = self.arr * other
            return RasterModel(arr=new_arr, raster_meta=self.raster_meta)
        elif isinstance(other, RasterModel):
            if self.raster_meta != other.raster_meta:
                msg = (
                    "Rasters must have the same metadata (e.g. CRS, cell size, etc.) "
                    "to be multiplied"
                )
                raise ValueError(msg)
            if self.arr.shape != other.arr.shape:
                msg = "Rasters must have the same shape to be multiplied"
                raise ValueError(msg)
            new_arr = self.arr * other.arr
            return RasterModel(arr=new_arr, raster_meta=self.raster_meta)
        else:
            return NotImplemented

    def __rmul__(self, other: float) -> Self:
        return self * other

    def __truediv__(self, other: float | Self) -> Self:
        if isinstance(other, float | int):
            new_arr = self.arr / other
            return RasterModel(arr=new_arr, raster_meta=self.raster_meta)
        elif isinstance(other, RasterModel):
            if self.raster_meta != other.raster_meta:
                msg = (
                    "Rasters must have the same metadata (e.g. CRS, cell size, etc.) "
                    "to be divided"
                )
                raise ValueError(msg)
            if self.arr.shape != other.arr.shape:
                msg = "Rasters must have the same shape to be divided"
                raise ValueError(msg)
            new_arr = self.arr / other.arr
            return RasterModel(arr=new_arr, raster_meta=self.raster_meta)
        else:
            return NotImplemented

    def __rtruediv__(self, other: float) -> Self:
        return self / other

    def __sub__(self, other: float | Self) -> Self:
        return self + (-other)

    def __rsub__(self, other: float) -> Self:
        return -self + other

    def __neg__(self) -> Self:
        return RasterModel(arr=-self.arr, raster_meta=self.raster_meta)

    @property
    def cell_centre_coords(self) -> NDArray[np.float64]:
        """Get the coordinates of the cell centres in the raster."""
        return self.raster_meta.get_cell_centre_coords(self.arr.shape)

    @contextmanager
    def to_rasterio_dataset(
        self,
    ) -> Generator[DatasetReader | BufferedDatasetWriter | DatasetWriter]:
        """Create a rasterio in-memory dataset from the Raster object.

        Example:
            >>> raster = RasterModel.example()
            >>> with raster.to_rasterio_dataset() as dataset:
            >>>     ...
        """
        memfile = MemoryFile()

        height, width = self.arr.shape

        try:
            with memfile.open(
                driver="GTiff",
                height=height,
                width=width,
                count=1,  # Assuming a single band; adjust as necessary
                dtype=self.arr.dtype,
                crs=self.raster_meta.crs.to_wkt(),
                transform=self.raster_meta.transform,
            ) as dataset:
                dataset.write(self.arr, 1)

            # Yield the dataset for reading
            with memfile.open() as dataset:
                yield dataset
        finally:
            memfile.close()

    def sample(
        self,
        xy: list[tuple[float, float]],
        *,
        na_action: Literal["raise", "ignore"] = "raise",
    ) -> NDArray[np.float64]:
        """Sample raster values at GeoSeries locations and return sampled values.

        Args:
            xy: A list of (x, y) coordinates to sample the raster at.
            na_action: Action to take when a NaN value is encountered in the input xy.
                       Options are "raise" (raise an error) or "ignore" (replace with
                       NaN).

        Returns:
            A list of sampled raster values for each geometry in the GeoSeries.
        """
        # If this function is too slow, consider the optimizations detailed here:
        # https://rdrn.me/optimising-sampling/

        # Short-circuit
        if len(xy) == 0:
            return np.array([], dtype=float)

        # Create in-memory rasterio dataset from the incumbent Raster object
        with self.to_rasterio_dataset() as dataset:
            if dataset.count != 1:
                msg = "Only single band rasters are supported."
                raise NotImplementedError(msg)

            xy_arr = np.array(xy)

            # Determine the indexes of any x,y coordinates where either is NaN.
            # We will drop these indexes for the purposes of calling .sample, but
            # then we will add NaN values back in at the end, inserting NaN into the
            # results array.
            xy_is_nan = np.isnan(xy_arr).any(axis=1)
            xy_nan_idxs = list(np.atleast_1d(np.squeeze(np.nonzero(xy_is_nan))))
            xy_arr = xy_arr[~xy_is_nan]

            if na_action == "raise" and len(xy_nan_idxs) > 0:
                nan_error_msg = "NaN value found in input coordinates"
                raise ValueError(nan_error_msg)

            # Sample the raster in-memory dataset (e.g. PGA values) at the coordinates
            samples = list(
                rasterio.sample.sample_gen(
                    dataset,
                    xy_arr,
                    indexes=1,  # Single band raster, N.B. rasterio is 1-indexed
                    masked=True,
                )
            )

            # Convert the sampled values to a NumPy array and set masked values to NaN
            raster_values = np.array(
                [s.data[0] if not s.mask else np.nan for s in samples]
            ).astype(float)

            if len(xy_nan_idxs) > 0:
                # Insert NaN values back into the results array
                # This is tricky because all the indexes get offset once we remove
                # elements.
                offset_xy_nan_idxs = xy_nan_idxs - np.arange(len(xy_nan_idxs))
                raster_values = np.insert(
                    raster_values,
                    offset_xy_nan_idxs,
                    np.nan,
                    axis=0,
                )

        return raster_values

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Bounding box of the raster as (xmin, ymin, xmax, ymax)"""
        x1, y1, x2, y2 = rasterio.transform.array_bounds(
            height=self.arr.shape[0],
            width=self.arr.shape[1],
            transform=self.raster_meta.transform,
        )
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        return (xmin, ymin, xmax, ymax)

    @property
    def bbox(self) -> Polygon:
        """Bounding box of the raster as a shapely polygon."""
        xmin, ymin, xmax, ymax = self.bounds
        return Polygon(
            [
                (xmin, ymin),
                (xmin, ymax),
                (xmax, ymax),
                (xmax, ymin),
                (xmin, ymin),
            ]
        )

    def explore(
        self,
        *,
        m: Map | None = None,
        opacity: float = 1.0,
        colormap: str = "viridis",
    ) -> Map:
        """Display the raster on a folium map."""
        if not FOLIUM_INSTALLED:
            msg = "The 'folium' package is required for 'explore()'."
            raise ImportError(msg)

        if m is None:
            m = folium.Map()

        rbga_map: Callable[[float], tuple[float, float, float, float]] = mpl.colormaps[
            colormap
        ]

        wgs84_crs = CRS.from_epsg(4326)
        gdf = gpd.GeoDataFrame(geometry=[self.bbox], crs=self.raster_meta.crs).to_crs(
            wgs84_crs
        )
        xmin, ymin, xmax, ymax = gdf.total_bounds

        arr = np.array(self.arr)

        # Normalize the data to the range [0, 1] as this is the cmap range
        min_val = np.nanmin(arr)
        max_val = np.nanmax(arr)
        if max_val > min_val:  # Prevent division by zero
            arr = (arr - min_val) / (max_val - min_val)
        else:
            arr = np.zeros_like(arr)  # In case all values are the same

        # Finally, need to determine whether to flip the image based on negative Affine
        # coefficients
        flip_x = self.raster_meta.transform.a < 0
        flip_y = self.raster_meta.transform.e > 0
        if flip_x:
            arr = np.flip(self.arr, axis=1)
        if flip_y:
            arr = np.flip(self.arr, axis=0)

        img = folium.raster_layers.ImageOverlay(
            image=arr,
            bounds=[[ymin, xmin], [ymax, xmax]],
            opacity=opacity,
            colormap=rbga_map,
            mercator_project=True,
        )

        img.add_to(m)

        m.fit_bounds([[ymin, xmin], [ymax, xmax]])

        return m

    def to_clipboard(self) -> None:
        """Copy the raster cell array to the clipboard."""
        pd.DataFrame(self.arr).to_clipboard(index=False, header=False)

    def plot(
        self,
        *,
        ax: Axes | None = None,
        cbar_label: str | None = None,
        basemap: bool = False,
        cmap: str = "viridis",
    ) -> Axes:
        """Plot the raster on a matplotlib axis."""
        if ax is None:
            _, ax = plt.subplots()
        ax: Axes

        if basemap:
            msg = "Basemap plotting is not yet implemented."
            raise NotImplementedError(msg)

        arr = self.arr.copy()

        # Get extent of the non-zero values in array index coordinates
        (x_nonzero,) = np.nonzero(arr.any(axis=0))
        (y_nonzero,) = np.nonzero(arr.any(axis=1))

        if len(x_nonzero) == 0 or len(y_nonzero) == 0:
            msg = "Raster contains no non-zero values; cannot plot."
            raise ValueError(msg)

        min_x_nonzero = np.min(x_nonzero)
        max_x_nonzero = np.max(x_nonzero)
        min_y_nonzero = np.min(y_nonzero)
        max_y_nonzero = np.max(y_nonzero)

        # Transform to raster CRS
        x1, y1 = self.raster_meta.transform * (min_x_nonzero, min_y_nonzero)
        x2, y2 = self.raster_meta.transform * (max_x_nonzero, max_y_nonzero)
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        arr[arr == 0] = np.nan

        with self.to_rasterio_dataset() as dataset:
            img, *_ = rasterio.plot.show(
                dataset, with_bounds=True, ax=ax, cmap=cmap
            ).get_images()

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_aspect("equal", "box")
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig = ax.get_figure()
        fig.colorbar(img, label=cbar_label, cax=cax)
        return ax

    def as_geodataframe(self, name: str = "value") -> gpd.GeoDataFrame:
        """Create a GeoDataFrame representation of the raster."""
        polygons = create_fishnet(bounds=self.bounds, res=self.raster_meta.cell_size)
        point_tuples = [polygon.centroid.coords[0] for polygon in polygons]
        raster_gdf = gpd.GeoDataFrame(
            {
                "geometry": polygons,
                name: self.sample(point_tuples, na_action="ignore"),
            },
            crs=self.raster_meta.crs,
        )

        return raster_gdf

    def to_file(self, path: Path) -> None:
        """Write the raster to a GeoTIFF file."""

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

        with rasterio.open(
            path,
            "w",
            driver=driver,
            height=self.arr.shape[0],
            width=self.arr.shape[1],
            count=1,
            dtype=self.arr.dtype,
            crs=self.raster_meta.crs,
            transform=self.raster_meta.transform,
            nodata=np.nan,
        ) as dst:
            try:
                dst.write(self.arr, 1)
            except CPLE_BaseError as err:
                msg = f"Failed to write raster to file: {err}"
                raise OSError(msg) from err

    def __str__(self) -> str:
        mean = np.nanmean(self.arr)
        return f"RasterModel(shape={self.arr.shape}, {mean=})"

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def example(cls) -> Self:
        """Create an example RasterModel."""
        # Peaks dataset style example
        n = 256
        x = np.linspace(-3, 3, n)
        y = np.linspace(-3, 3, n)
        x, y = np.meshgrid(x, y)
        z = np.exp(-(x**2) - y**2) * np.sin(3 * np.sqrt(x**2 + y**2))
        arr = z.astype(np.float32)

        raster_meta = RasterMeta.example()
        return cls(arr=arr, raster_meta=raster_meta)

    def fillna(self, value: float) -> Self:
        """Fill NaN values in the raster with a specified value.

        See also `extrapolate()` for filling NaN values using extrapolation from data.
        """
        filled_arr = np.nan_to_num(self.arr, nan=value)
        new_raster = self.model_copy()
        new_raster.arr = filled_arr
        return new_raster

    def get_xy(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the x and y coordinates of the raster in meshgrid format."""
        col_idx, row_idx = np.meshgrid(
            np.arange(self.arr.shape[1]),
            np.arange(self.arr.shape[0]),
        )

        col_idx = col_idx.flatten()
        row_idx = row_idx.flatten()

        coords = np.vstack((row_idx, col_idx)).T

        x, y = rasterio.transform.xy(self.raster_meta.transform, *coords.T)
        x = np.array(x).reshape(self.arr.shape)
        y = np.array(y).reshape(self.arr.shape)
        return x, y

    def contour(
        self, *, levels: list[float], smoothing: bool = True
    ) -> gpd.GeoDataFrame:
        """Create contour lines from the raster data, optionally with smoothing.

        The contour lines are returned as a GeoDataFrame with the contours as linestring
        geometries and the contour levels as attributes in a column named 'level'.

        Consider calling `blur()` before this method to smooth the raster data before
        contouring, to denoise the contours.

        Args:
            levels: A list of contour levels to generate. The contour lines will be
                    generated for each level in this list.
            smoothing: Defaults to true, which corresponds to applying a smoothing
                       algorithm to the contour lines. At the moment, this is the
                       Catmull-Rom spline algorithm. If set to False, the raw
                       contours will be returned without any smoothing.
        """

        all_levels = []
        all_geoms = []
        for level in levels:
            contours = skimage.measure.find_contours(
                self.arr,
                level=level,
            )

            # Constructg shapely LineString objects
            # Convert to CRS from array index coordinates to raster CRS
            geoms = [
                LineString(
                    np.array(
                        rasterio.transform.xy(self.raster_meta.transform, *contour.T)
                    ).T
                )
                for contour in contours
                # Contour lines need at least three distinct points to avoid
                # degenerate geometries
                if np.unique(contour, axis=0).shape[0] > 2
            ]

            # Apply smoothing if requested
            if smoothing:
                geoms = [catmull_rom_smooth(geom) for geom in geoms]

            all_geoms.extend(geoms)
            all_levels.extend([level] * len(geoms))

        contour_gdf = gpd.GeoDataFrame(
            data={
                "level": all_levels,
            },
            geometry=all_geoms,
            crs=self.raster_meta.crs,
        )

        return contour_gdf

    def blur(self, sigma: float) -> Self:
        """Apply a Gaussian blur to the raster data.

        Args:
            sigma: Standard deviation for Gaussian kernel, in units of geographic
                   coordinate distance (e.g. meters). A larger sigma results in a more
                   blurred image.
        """

        cell_sigma = sigma / self.raster_meta.cell_size

        blurred_array = gaussian_filter(self.arr, sigma=cell_sigma)
        new_raster = self.model_copy()
        new_raster.arr = blurred_array
        return new_raster

    def extrapolate(self, method: Literal["nearest"] = "nearest") -> Self:
        """Extrapolate the raster data to fill NaN values.

        See also `fillna()` for filling NaN values with a specific value.

        If the raster is all-NaN, this method will return a copy of the raster without
        changing the NaN values.

        Args:
            method: The method to use for extrapolation. Currently only 'nearest' is
                    supported, which fills NaN values with the nearest non-NaN value.
        """
        if method not in ("nearest",):
            msg = f"Unsupported extrapolation method: {method}"
            raise NotImplementedError(msg)

        raster = self.model_copy()
        raster.arr = fillna_nearest_neighbours(arr=self.arr)

        return raster

    def resample(
        self, new_cell_size: float, *, method: Literal["bilinear"] = "bilinear"
    ) -> Self:
        """Resample the raster data to a new resolution.

        If the new cell size is not an exact multiple of the current cell size, the
        overall raster bounds may increase slightly. The affine transform will keep
        the same shift, i.e. the top-left corner of the raster will remain in the same'
        coordinate location. A corollary is that the overall centre of the raster bounds
        will not necessary be the same as the original raster.

        Args:
            new_cell_size: The desired cell size for the resampled raster.
            method: The resampling method to use. Only 'bilinear' is supported.
        """
        if method not in ("bilinear",):
            msg = f"Unsupported resampling method: {method}"
            raise NotImplementedError(msg)

        factor = self.raster_meta.cell_size / new_cell_size

        # Use the rasterio dataset with proper context management
        with self.to_rasterio_dataset() as dataset:
            # N.B. the new height and width may increase slightly.
            new_height = int(np.ceil(dataset.height * factor))
            new_width = int(np.ceil(dataset.width * factor))

            # Resample via rasterio
            (new_arr,) = dataset.read(  # Assume exactly one band
                out_shape=(dataset.count, new_height, new_width),
                resampling=Resampling.bilinear,
            )

            # Create new RasterMeta with updated transform and cell size
            new_raster_meta = RasterMeta(
                transform=dataset.transform
                * dataset.transform.scale(
                    (dataset.width / new_width),
                    (dataset.height / new_height),
                ),
                crs=self.raster_meta.crs,
                cell_size=new_cell_size,
            )

            return RasterModel(arr=new_arr, raster_meta=new_raster_meta)

    @field_validator("arr")
    @classmethod
    def check_2d_array(cls, v: np.ndarray) -> np.ndarray:
        """Validator to ensure the cell array is 2D."""
        if v.ndim != 2:
            msg = "Cell array must be 2D"
            raise RasterCellArrayShapeError(msg)
        return v
