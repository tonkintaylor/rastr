"""Raster data structure."""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Collection
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import numpy.ma
import rasterio.plot
import rasterio.sample
import rasterio.transform
import skimage.measure
from affine import Affine
from pydantic import BaseModel, InstanceOf, field_validator
from pyproj.crs.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from shapely.geometry import LineString, Point, Polygon

from rastr.arr.fill import fillna_nearest_neighbours
from rastr.gis.fishnet import create_fishnet
from rastr.gis.smooth import catmull_rom_smooth
from rastr.meta import RasterMeta

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import geopandas as gpd
    from folium import Map
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray
    from rasterio.io import BufferedDatasetWriter, DatasetReader, DatasetWriter
    from typing_extensions import Self

try:
    from rasterio._err import CPLE_BaseError
except ImportError:
    CPLE_BaseError = Exception  # Fallback if private module import fails


FOLIUM_INSTALLED = importlib.util.find_spec("folium") is not None
BRANCA_INSTALLED = importlib.util.find_spec("branca") is not None
MATPLOTLIB_INSTALLED = importlib.util.find_spec("matplotlib") is not None

CONTOUR_PERTURB_EPS = 1e-10


class RasterCellArrayShapeError(ValueError):
    """Custom error for invalid raster cell array shapes."""


class RasterModel(BaseModel):
    """2-dimensional raster and metadata."""

    arr: InstanceOf[np.ndarray]
    raster_meta: RasterMeta

    @property
    def meta(self) -> RasterMeta:
        """Alias for raster_meta."""
        return self.raster_meta

    @meta.setter
    def meta(self, value: RasterMeta) -> None:
        self.raster_meta = value

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the raster array."""
        return self.arr.shape

    @property
    def crs(self) -> CRS:
        """Convenience property to access the CRS via meta."""
        return self.meta.crs

    @crs.setter
    def crs(self, value: CRS) -> None:
        """Set the CRS via meta."""
        self.meta.crs = value

    @property
    def transform(self) -> Affine:
        """Convenience property to access the transform via meta."""
        return self.meta.transform

    @transform.setter
    def transform(self, value: Affine) -> None:
        """Set the transform via meta."""
        self.meta.transform = value

    def __init__(
        self,
        *,
        arr: ArrayLike,
        meta: RasterMeta | None = None,
        raster_meta: RasterMeta | None = None,
    ) -> None:
        arr = np.asarray(arr)

        # Set the meta
        if meta is not None and raster_meta is not None:
            msg = (
                "Only one of 'meta' or 'raster_meta' should be provided, they are "
                "aliases."
            )
            raise ValueError(msg)
        elif meta is not None and raster_meta is None:
            raster_meta = meta
        elif meta is None and raster_meta is not None:
            pass
        else:
            # Don't need to mention `'meta'` to simplify the messaging.
            msg = "The attribute 'raster_meta' is required."
            raise ValueError(msg)

        super().__init__(arr=arr, raster_meta=raster_meta)

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
        cls = self.__class__
        if isinstance(other, float | int):
            new_arr = self.arr + other
            return cls(arr=new_arr, raster_meta=self.raster_meta)
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
            return cls(arr=new_arr, raster_meta=self.raster_meta)
        else:
            return NotImplemented

    def __radd__(self, other: float) -> Self:
        return self + other

    def __mul__(self, other: float | Self) -> Self:
        cls = self.__class__
        if isinstance(other, float | int):
            new_arr = self.arr * other
            return cls(arr=new_arr, raster_meta=self.raster_meta)
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
            return cls(arr=new_arr, raster_meta=self.raster_meta)
        else:
            return NotImplemented

    def __rmul__(self, other: float) -> Self:
        return self * other

    def __truediv__(self, other: float | Self) -> Self:
        cls = self.__class__
        if isinstance(other, float | int):
            new_arr = self.arr / other
            return cls(arr=new_arr, raster_meta=self.raster_meta)
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
            return cls(arr=new_arr, raster_meta=self.raster_meta)
        else:
            return NotImplemented

    def __rtruediv__(self, other: float) -> Self:
        return self / other

    def __sub__(self, other: float | Self) -> Self:
        return self + (-other)

    def __rsub__(self, other: float) -> Self:
        return -self + other

    def __neg__(self) -> Self:
        cls = self.__class__
        return cls(arr=-self.arr, raster_meta=self.raster_meta)

    @property
    def cell_centre_coords(self) -> NDArray[np.float64]:
        """Get the coordinates of the cell centres in the raster."""
        return self.raster_meta.get_cell_centre_coords(self.arr.shape)

    @property
    def cell_x_coords(self) -> NDArray[np.float64]:
        """Get the x coordinates of the cell centres in the raster."""
        return self.raster_meta.get_cell_x_coords(self.arr.shape[1])

    @property
    def cell_y_coords(self) -> NDArray[np.float64]:
        """Get the y coordinates of the cell centres in the raster."""
        return self.raster_meta.get_cell_y_coords(self.arr.shape[0])

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

    @overload
    def sample(
        self,
        xy: Collection[tuple[float, float]] | Collection[Point] | ArrayLike,
        *,
        na_action: Literal["raise", "ignore"] = "raise",
    ) -> NDArray: ...
    @overload
    def sample(
        self,
        xy: tuple[float, float] | Point,
        *,
        na_action: Literal["raise", "ignore"] = "raise",
    ) -> float: ...
    def sample(
        self,
        xy: Collection[tuple[float, float]]
        | Collection[Point]
        | ArrayLike
        | tuple[float, float]
        | Point,
        *,
        na_action: Literal["raise", "ignore"] = "raise",
    ) -> NDArray | float:
        """Sample raster values at GeoSeries locations and return sampled values.

        Args:
            xy: A list of (x, y) coordinates or shapely Point objects to sample the
                raster at.
            na_action: Action to take when a NaN value is encountered in the input xy.
                       Options are "raise" (raise an error) or "ignore" (replace with
                       NaN).

        Returns:
            A list of sampled raster values for each geometry in the GeoSeries.
        """
        # If this function is too slow, consider the optimizations detailed here:
        # https://rdrn.me/optimising-sampling/

        # Convert shapely Points to coordinate tuples if needed
        if isinstance(xy, Point):
            xy = [(xy.x, xy.y)]
            singleton = True
        elif (
            isinstance(xy, Collection)
            and len(xy) > 0
            and isinstance(next(iter(xy)), Point)
        ):
            xy = [(point.x, point.y) for point in xy]  # pyright: ignore[reportAttributeAccessIssue]
            singleton = False
        elif (
            isinstance(xy, tuple)
            and len(xy) == 2
            and isinstance(next(iter(xy)), (float, int))
        ):
            xy = [xy]  # pyright: ignore[reportAssignmentType]
            singleton = True
        else:
            singleton = False

        xy = np.asarray(xy, dtype=float)

        if len(xy) == 0:
            # Short-circuit
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
                [s.data[0] if not numpy.ma.getmask(s) else np.nan for s in samples]
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

        if singleton:
            (raster_value,) = raster_values
            return raster_value

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
        cbar_label: str | None = None,
    ) -> Map:
        """Display the raster on a folium map."""
        if not FOLIUM_INSTALLED or not MATPLOTLIB_INSTALLED:
            msg = "The 'folium' and 'matplotlib' packages are required for 'explore()'."
            raise ImportError(msg)

        import folium.raster_layers
        import geopandas as gpd
        import matplotlib as mpl

        if m is None:
            m = folium.Map()

        rgba_map: Callable[[float], tuple[float, float, float, float]] = mpl.colormaps[
            colormap
        ]

        # Cast to GDF to facilitate converting bounds to WGS84
        wgs84_crs = CRS.from_epsg(4326)
        gdf = gpd.GeoDataFrame(geometry=[self.bbox], crs=self.raster_meta.crs).to_crs(
            wgs84_crs
        )
        xmin, ymin, xmax, ymax = gdf.total_bounds

        arr = np.array(self.arr)

        # Normalize the data to the range [0, 1] as this is the cmap range
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="All-NaN slice encountered",
                category=RuntimeWarning,
            )
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
            arr = np.flip(arr, axis=1)
        if flip_y:
            arr = np.flip(arr, axis=0)

        img = folium.raster_layers.ImageOverlay(
            image=arr,
            bounds=[[ymin, xmin], [ymax, xmax]],
            opacity=opacity,
            colormap=rgba_map,
            mercator_project=True,
        )

        img.add_to(m)

        # Add a colorbar legend
        if BRANCA_INSTALLED:
            from branca.colormap import LinearColormap as BrancaLinearColormap
            from matplotlib.colors import to_hex

            # Determine legend data range in original units
            vmin = float(min_val) if np.isfinite(min_val) else 0.0
            vmax = float(max_val) if np.isfinite(max_val) else 1.0
            if vmax <= vmin:
                vmax = vmin + 1.0

            sample_points = np.linspace(0, 1, rgba_map.N)
            colors = [to_hex(rgba_map(x)) for x in sample_points]
            legend = BrancaLinearColormap(colors=colors, vmin=vmin, vmax=vmax)
            if cbar_label:
                legend.caption = cbar_label
            legend.add_to(m)

        m.fit_bounds([[ymin, xmin], [ymax, xmax]])

        return m

    def to_clipboard(self) -> None:
        """Copy the raster cell array to the clipboard."""
        import pandas as pd

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
        if not MATPLOTLIB_INSTALLED:
            msg = "The 'matplotlib' package is required for 'plot()'."
            raise ImportError(msg)

        from matplotlib import pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if ax is None:
            _, _ax = plt.subplots()
            _ax: Axes
            ax = _ax

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
        x1, y1 = self.raster_meta.transform * (min_x_nonzero, min_y_nonzero)  # type: ignore[reportAssignmentType] overloaded tuple size in affine
        x2, y2 = self.raster_meta.transform * (max_x_nonzero, max_y_nonzero)  # type: ignore[reportAssignmentType]
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
        if fig is not None:
            fig.colorbar(img, label=cbar_label, cax=cax)
        return ax

    def as_geodataframe(self, name: str = "value") -> gpd.GeoDataFrame:
        """Create a GeoDataFrame representation of the raster."""
        import geopandas as gpd

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

    def to_file(self, path: Path | str) -> None:
        """Write the raster to a GeoTIFF file."""

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
        cls = self.__class__
        mean = np.nanmean(self.arr)
        return f"{cls.__name__}(shape={self.arr.shape}, {mean=})"

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

    @overload
    def apply(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        *,
        raw: Literal[True],
    ) -> Self: ...
    @overload
    def apply(
        self,
        func: Callable[[float], float] | Callable[[np.ndarray], np.ndarray],
        *,
        raw: Literal[False] = False,
    ) -> Self: ...
    def apply(self, func, *, raw=False) -> Self:
        """Apply a function element-wise to the raster array.

        Creates a new raster instance with the same metadata (CRS, transform, etc.)
        but with the data array transformed by the provided function. The original
        raster is not modified.

        Args:
            func: The function to apply to the raster array. If `raw` is True, this
                  function should accept and return a NumPy array. If `raw` is False,
                  this function should accept and return a single float value.
            raw: If True, the function is applied directly to the entire array at
                 once. If False, the function is applied element-wise to each cell
                 in the array using `np.vectorize()`. Default is False.
        """
        new_raster = self.model_copy()
        if raw:
            new_arr = func(self.arr)
        else:
            new_arr = np.vectorize(func)(self.arr)
        new_raster.arr = np.asarray(new_arr)
        return new_raster

    def fillna(self, value: float) -> Self:
        """Fill NaN values in the raster with a specified value.

        See also `extrapolate()` for filling NaN values using extrapolation from data.
        """
        filled_arr = np.nan_to_num(self.arr, nan=value)
        new_raster = self.model_copy()
        new_raster.arr = filled_arr
        return new_raster

    def get_xy(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the x and y coordinates of the raster cell centres in meshgrid format.

        Returns the coordinates of the cell centres as two separate 2D arrays in
        meshgrid format, where each array has the same shape as the raster data array.

        Returns:
            A tuple of (x, y) coordinate arrays where:
            - x: 2D array of x-coordinates of cell centres
            - y: 2D array of y-coordinates of cell centres
            Both arrays have the same shape as the raster data array.
        """
        coords = self.raster_meta.get_cell_centre_coords(self.arr.shape)
        return coords[:, :, 0], coords[:, :, 1]

    def contour(
        self, levels: list[float] | NDArray, *, smoothing: bool = True
    ) -> gpd.GeoDataFrame:
        """Create contour lines from the raster data, optionally with smoothing.

        The contour lines are returned as a GeoDataFrame with the contours dissolved
        by level, resulting in one row per contour level. Each row contains a
        (Multi)LineString geometry representing all contour lines for that level,
        and the contour level value in a column named 'level'.

        Consider calling `blur()` before this method to smooth the raster data before
        contouring, to denoise the contours.

        Args:
            levels: A list or array of contour levels to generate. The contour lines
                    will be generated for each level in this sequence.
            smoothing: Defaults to true, which corresponds to applying a smoothing
                       algorithm to the contour lines. At the moment, this is the
                       Catmull-Rom spline algorithm. If set to False, the raw
                       contours will be returned without any smoothing.
        """
        import geopandas as gpd

        all_levels = []
        all_geoms = []
        arr_max = np.nanmax(self.arr)
        arr_min = np.nanmin(self.arr)
        for level in levels:
            # If this is the maximum or minimum level, perturb it ever-so-slightly to
            # ensure we get contours at the edges of the raster
            perturbed_level = level
            if level == arr_max:
                perturbed_level -= CONTOUR_PERTURB_EPS
            elif level == arr_min:
                perturbed_level += CONTOUR_PERTURB_EPS

            contours = skimage.measure.find_contours(
                self.arr,
                level=perturbed_level,
            )

            # Construct shapely LineString objects
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

        # Dissolve contours by level to merge all contour lines of the same level
        return contour_gdf.dissolve(by="level", as_index=False)

    def blur(self, sigma: float) -> Self:
        """Apply a Gaussian blur to the raster data.

        Args:
            sigma: Standard deviation for Gaussian kernel, in units of geographic
                   coordinate distance (e.g. meters). A larger sigma results in a more
                   blurred image.
        """
        from scipy.ndimage import gaussian_filter

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

    def crop(
        self,
        bounds: tuple[float, float, float, float],
        *,
        strategy: Literal["underflow", "overflow"] = "underflow",
    ) -> Self:
        """Crop the raster to the specified bounds as (minx, miny, maxx, maxy).

        Args:
            bounds: A tuple of (minx, miny, maxx, maxy) defining the bounds to crop to.
            strategy:   The cropping strategy to use. 'underflow' will crop the raster
                        to be fully within the bounds, ignoring any cells that are
                        partially outside the bounds. 'overflow' will instead include
                        cells that intersect the bounds, ensuring the bounds area
                        remains covered with cells.

        Returns:
            A new RasterModel instance cropped to the specified bounds.
        """

        minx, miny, maxx, maxy = bounds
        arr = self.arr

        # Get the half cell size for cropping
        cell_size = self.raster_meta.cell_size
        half_cell_size = cell_size / 2

        # Get the cell centre coordinates as 1D arrays
        x_coords = self.cell_x_coords
        y_coords = self.cell_y_coords

        # Get the indices to crop the array
        if strategy == "underflow":
            x_idx = (x_coords >= minx + half_cell_size) & (
                x_coords <= maxx - half_cell_size
            )
            y_idx = (y_coords >= miny + half_cell_size) & (
                y_coords <= maxy - half_cell_size
            )
        elif strategy == "overflow":
            x_idx = (x_coords > minx - half_cell_size) & (
                x_coords < maxx + half_cell_size
            )
            y_idx = (y_coords > miny - half_cell_size) & (
                y_coords < maxy + half_cell_size
            )
        else:
            msg = f"Unsupported cropping strategy: {strategy}"
            raise NotImplementedError(msg)

        # Crop the array
        cropped_arr = arr[np.ix_(y_idx, x_idx)]

        # Check the shape of the cropped array
        if cropped_arr.size == 0:
            msg = "Cropped array is empty; no cells within the specified bounds."
            raise ValueError(msg)

        # Recalculate the transform for the cropped raster
        x_coords = x_coords[x_idx]
        y_coords = y_coords[y_idx]
        transform = rasterio.transform.from_bounds(
            west=x_coords.min() - half_cell_size,
            south=y_coords.min() - half_cell_size,
            east=x_coords.max() + half_cell_size,
            north=y_coords.max() + half_cell_size,
            width=cropped_arr.shape[1],
            height=cropped_arr.shape[0],
        )

        # Update the raster
        cls = self.__class__
        new_meta = RasterMeta(
            cell_size=cell_size, crs=self.raster_meta.crs, transform=transform
        )
        return cls(arr=cropped_arr, raster_meta=new_meta)

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

        cls = self.__class__
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

            return cls(arr=new_arr, raster_meta=new_raster_meta)

    @field_validator("arr")
    @classmethod
    def check_2d_array(cls, v: NDArray) -> NDArray:
        """Validator to ensure the cell array is 2D."""
        if v.ndim != 2:
            msg = "Cell array must be 2D"
            raise RasterCellArrayShapeError(msg)
        return v
