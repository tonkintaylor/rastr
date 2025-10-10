"""Raster data structure."""

from __future__ import annotations

import importlib.util
import warnings
from collections.abc import Collection
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import numpy.ma
import rasterio.features
import rasterio.plot
import rasterio.sample
import rasterio.transform
import skimage.measure
from pydantic import BaseModel, InstanceOf, field_validator
from pyproj import Transformer
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
    from affine import Affine
    from branca.colormap import LinearColormap as BrancaLinearColormap
    from folium import Map
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage
    from numpy.typing import ArrayLike, NDArray
    from rasterio.io import BufferedDatasetWriter, DatasetReader, DatasetWriter
    from shapely import MultiPolygon
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


class Raster(BaseModel):
    """2-dimensional raster and metadata."""

    arr: InstanceOf[np.ndarray]
    raster_meta: RasterMeta

    @field_validator("arr")
    @classmethod
    def check_2d_array(cls, v: NDArray) -> NDArray:
        """Validator to ensure the cell array is 2D."""
        if v.ndim != 2:
            msg = "Cell array must be 2D"
            raise RasterCellArrayShapeError(msg)
        return v

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

    @property
    def cell_size(self) -> float:
        """Convenience property to access the cell size via meta."""
        return self.meta.cell_size

    @cell_size.setter
    def cell_size(self, value: float) -> None:
        """Set the cell size via meta."""
        self.meta.cell_size = value

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
        """Check equality of two Raster objects."""
        if not isinstance(other, Raster):
            return NotImplemented
        return (
            np.array_equal(self.arr, other.arr)
            and self.raster_meta == other.raster_meta
        )

    def is_like(self, other: Raster) -> bool:
        """Check if two Raster objects have the same metadata and shape.

        Args:
            other: Another Raster to compare with.

        Returns:
            True if both rasters have the same meta and shape attributes.
        """
        return self.meta == other.meta and self.shape == other.shape

    __hash__ = BaseModel.__hash__

    def __add__(self, other: float | Self) -> Self:
        cls = self.__class__
        if isinstance(other, float | int):
            new_arr = self.arr + other
            return cls(arr=new_arr, raster_meta=self.raster_meta)
        elif isinstance(other, Raster):
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
        elif isinstance(other, Raster):
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
        elif isinstance(other, Raster):
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
            >>> raster = Raster.example()
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

    def explore(  # noqa: PLR0913 c.f. geopandas.explore which also has many input args
        self,
        *,
        m: Map | None = None,
        opacity: float = 1.0,
        colormap: str
        | Callable[[float], tuple[float, float, float, float]] = "viridis",
        cbar_label: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> Map:
        """Display the raster on a folium map."""
        if not FOLIUM_INSTALLED or not MATPLOTLIB_INSTALLED:
            msg = "The 'folium' and 'matplotlib' packages are required for 'explore()'."
            raise ImportError(msg)

        import folium.raster_layers
        import matplotlib as mpl

        if m is None:
            m = folium.Map()

        if vmin is not None and vmax is not None and vmax <= vmin:
            msg = "'vmin' must be less than 'vmax'."
            raise ValueError(msg)

        if isinstance(colormap, str):
            colormap = mpl.colormaps[colormap]

        # Transform bounds to WGS84 using pyproj directly
        wgs84_crs = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(
            self.raster_meta.crs, wgs84_crs, always_xy=True
        )

        # Get the corner points of the bounding box
        raster_xmin, raster_ymin, raster_xmax, raster_ymax = self.bounds
        corner_points = [
            (raster_xmin, raster_ymin),
            (raster_xmin, raster_ymax),
            (raster_xmax, raster_ymax),
            (raster_xmax, raster_ymin),
        ]

        # Transform all corner points to WGS84
        transformed_points = [transformer.transform(x, y) for x, y in corner_points]

        # Find the bounding box of the transformed points
        transformed_xs, transformed_ys = zip(*transformed_points, strict=True)
        xmin, xmax = min(transformed_xs), max(transformed_xs)
        ymin, ymax = min(transformed_ys), max(transformed_ys)

        # Normalize the array to [0, 1] for colormap mapping
        _vmin, _vmax = _get_vmin_vmax(self, vmin=vmin, vmax=vmax)
        arr = self.normalize(vmin=_vmin, vmax=_vmax).arr

        # Finally, need to determine whether to flip the image based on negative Affine
        # coefficients
        flip_x = self.raster_meta.transform.a < 0
        flip_y = self.raster_meta.transform.e > 0
        if flip_x:
            arr = np.flip(arr, axis=1)
        if flip_y:
            arr = np.flip(arr, axis=0)

        bounds = [[ymin, xmin], [ymax, xmax]]
        img = folium.raster_layers.ImageOverlay(
            image=arr,
            bounds=bounds,
            opacity=opacity,
            colormap=colormap,
            mercator_project=True,
        )

        img.add_to(m)

        # Add a colorbar legend
        if BRANCA_INSTALLED:
            cbar = _map_colorbar(colormap=colormap, vmin=_vmin, vmax=_vmax)
            if cbar_label:
                cbar.caption = cbar_label
            cbar.add_to(m)

        m.fit_bounds(bounds)

        return m

    def normalize(
        self, *, vmin: float | None = None, vmax: float | None = None
    ) -> Self:
        """Normalize the raster values to the range [0, 1].

        If custom vmin and vmax values are provided, values below vmin will be set to 0,
        and values above vmax will be set to 1.

        Args:
            vmin: Minimum value for normalization. Values below this will be set to 0.
                  If None, the minimum value in the array is used.
            vmax: Maximum value for normalization. Values above this will be set to 1.
                  If None, the maximum value in the array is used.
        """
        _vmin, _vmax = _get_vmin_vmax(self, vmin=vmin, vmax=vmax)

        arr = self.arr.copy()
        if _vmax > _vmin:
            arr = (arr - _vmin) / (_vmax - _vmin)
            arr = np.clip(arr, 0, 1)
        else:
            arr = np.zeros_like(arr)
        return self.__class__(arr=arr, raster_meta=self.raster_meta)

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
        suppressed: Collection[float] | float = tuple(),
        **kwargs: Any,
    ) -> Axes:
        """Plot the raster on a matplotlib axis.

        Args:
            ax: A matplotlib axes object to plot on. If None, a new figure will be
                created.
            cbar_label: Label for the colorbar. If None, no label is added.
            basemap: Whether to add a basemap. Currently not implemented.
            cmap: Colormap to use for the plot.
            suppressed: Values to suppress from the plot (i.e. not display). This can be
                        useful for zeroes especially.
            **kwargs: Additional keyword arguments to pass to `rasterio.plot.show()`.
                      This includes parameters like `alpha` for transparency.
        """
        if not MATPLOTLIB_INSTALLED:
            msg = "The 'matplotlib' package is required for 'plot()'."
            raise ImportError(msg)

        from matplotlib import pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        suppressed = np.array(suppressed)

        if ax is None:
            _, _ax = plt.subplots()
            _ax: Axes
            ax = _ax

        if basemap:
            msg = "Basemap plotting is not yet implemented."
            raise NotImplementedError(msg)

        model = self.model_copy()
        model.arr = model.arr.copy()

        # Get extent of the unsuppressed values in array index coordinates
        suppressed_mask = np.isin(model.arr, suppressed)
        (x_unsuppressed,) = np.nonzero((~suppressed_mask).any(axis=0))
        (y_unsuppressed,) = np.nonzero((~suppressed_mask).any(axis=1))

        if len(x_unsuppressed) == 0 or len(y_unsuppressed) == 0:
            msg = "Raster contains no unsuppressed values; cannot plot."
            raise ValueError(msg)

        # N.B. these are array index coordinates, so np.min and np.max are safe since
        # they cannot encounter NaN values.
        min_x_unsuppressed = np.min(x_unsuppressed)
        max_x_unsuppressed = np.max(x_unsuppressed)
        min_y_unsuppressed = np.min(y_unsuppressed)
        max_y_unsuppressed = np.max(y_unsuppressed)

        # Transform to raster CRS
        x1, y1 = self.raster_meta.transform * (min_x_unsuppressed, min_y_unsuppressed)  # type: ignore[reportAssignmentType] overloaded tuple size in affine
        x2, y2 = self.raster_meta.transform * (max_x_unsuppressed, max_y_unsuppressed)  # type: ignore[reportAssignmentType]
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        model.arr[suppressed_mask] = np.nan

        img, *_ = model.rio_show(ax=ax, cmap=cmap, with_bounds=True, **kwargs)

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

    def rio_show(self, **kwargs: Any) -> list[AxesImage]:
        """Plot the raster using rasterio's built-in plotting function.

        This is useful for lower-level access to rasterio's plotting capabilities.
        Generally, the `plot()` method is preferred for most use cases.

        Args:
            **kwargs: Keyword arguments to pass to `rasterio.plot.show()`. This includes
            parameters like `alpha` for transparency, and `with_bounds` to control
            whether to plot in spatial coordinates or array index coordinates.
        """
        with self.to_rasterio_dataset() as dataset:
            return rasterio.plot.show(dataset, **kwargs).get_images()

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

    def to_file(self, path: Path | str, **kwargs: Any) -> None:
        """Write the raster to a GeoTIFF file.

        Args:
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
            arr_to_write = np.where(np.isnan(self.arr), nodata_value, self.arr)
        else:
            nodata_value = np.nan
            arr_to_write = self.arr

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
            nodata=nodata_value,
            **kwargs,
        ) as dst:
            try:
                dst.write(arr_to_write, 1)
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
        """Create an example Raster."""
        # Peaks dataset style example
        n = 256
        x = np.linspace(-3, 3, n)
        y = np.linspace(-3, 3, n)
        x, y = np.meshgrid(x, y)
        z = np.exp(-(x**2) - y**2) * np.sin(3 * np.sqrt(x**2 + y**2))
        arr = z.astype(np.float32)

        raster_meta = RasterMeta.example()
        return cls(arr=arr, raster_meta=raster_meta)

    @classmethod
    def full_like(cls, other: Raster, *, fill_value: float) -> Self:
        """Create a raster with the same metadata as another but filled with a constant.

        Args:
            other: The raster to copy metadata from.
            fill_value: The constant value to fill all cells with.

        Returns:
            A new raster with the same shape and metadata as `other`, but with all cells
            set to `fill_value`.
        """
        arr = np.full(other.shape, fill_value, dtype=np.float32)
        return cls(arr=arr, raster_meta=other.raster_meta)

    @classmethod
    def read_file(cls, filename: Path | str, crs: CRS | str | None = None) -> Self:
        """Read raster data from a file and return an in-memory Raster object.

        Args:
            filename: Path to the raster file.
            crs: Optional coordinate reference system to override the file's CRS.
        """
        # Import here to avoid circular import (rastr.io imports Raster)
        from rastr.io import read_raster_inmem  # noqa: PLC0415

        return read_raster_inmem(filename, crs=crs, cls=cls)

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

    def max(self) -> float:
        """Get the maximum value in the raster, ignoring NaN values.

        Returns:
            The maximum value in the raster. Returns NaN if all values are NaN.
        """
        return float(np.nanmax(self.arr))

    def min(self) -> float:
        """Get the minimum value in the raster, ignoring NaN values.

        Returns:
            The minimum value in the raster. Returns NaN if all values are NaN.
        """
        return float(np.nanmin(self.arr))

    def mean(self) -> float:
        """Get the mean value in the raster, ignoring NaN values.

        Returns:
            The mean value in the raster. Returns NaN if all values are NaN.
        """
        return float(np.nanmean(self.arr))

    def std(self) -> float:
        """Get the standard deviation of values in the raster, ignoring NaN values.

        Returns:
            The standard deviation of the raster. Returns NaN if all values are NaN.
        """
        return float(np.nanstd(self.arr))

    def quantile(self, q: float) -> float:
        """Get the specified quantile value in the raster, ignoring NaN values.

        Args:
            q: Quantile to compute, must be between 0 and 1 inclusive.

        Returns:
            The quantile value. Returns NaN if all values are NaN.
        """
        return float(np.nanquantile(self.arr, q))

    def median(self) -> float:
        """Get the median value in the raster, ignoring NaN values.

        This is equivalent to quantile(0.5).

        Returns:
            The median value in the raster. Returns NaN if all values are NaN.
        """
        return float(np.nanmedian(self.arr))

    def fillna(self, value: float) -> Self:
        """Fill NaN values in the raster with a specified value.

        See also `extrapolate()` for filling NaN values using extrapolation from data.
        """
        filled_arr = np.nan_to_num(self.arr, nan=value)
        new_raster = self.model_copy()
        new_raster.arr = filled_arr
        return new_raster

    def copy(self) -> Self:  # type: ignore[override]
        """Create a copy of the raster.

        This method wraps `model_copy()` for convenience.

        Returns:
            A new Raster instance.
        """
        return self.model_copy(deep=True)

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
        self, levels: Collection[float] | NDArray, *, smoothing: bool = True
    ) -> gpd.GeoDataFrame:
        """Create contour lines from the raster data, optionally with smoothing.

        The contour lines are returned as a GeoDataFrame with the contours dissolved
        by level, resulting in one row per contour level. Each row contains a
        (Multi)LineString geometry representing all contour lines for that level,
        and the contour level value in a column named 'level'.

        Consider calling `blur()` before this method to smooth the raster data before
        contouring, to denoise the contours.

        Args:
            levels: A collection or array of contour levels to generate. The contour
                    lines will be generated for each level in this sequence.
            smoothing: Defaults to true, which corresponds to applying a smoothing
                       algorithm to the contour lines. At the moment, this is the
                       Catmull-Rom spline algorithm. If set to False, the raw
                       contours will be returned without any smoothing.
        """
        import geopandas as gpd

        all_levels = []
        all_geoms = []
        for level in levels:
            # If this is the maximum or minimum level, perturb it ever-so-slightly to
            # ensure we get contours at the edges of the raster
            perturbed_level = level
            if level == self.max():
                perturbed_level -= CONTOUR_PERTURB_EPS
            elif level == self.min():
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

    def blur(self, sigma: float, *, preserve_nan: bool = True) -> Self:
        """Apply a Gaussian blur to the raster data.

        Args:
            sigma: Standard deviation for Gaussian kernel, in units of geographic
                   coordinate distance (e.g. meters). A larger sigma results in a more
                   blurred image.
            preserve_nan: If True, applies NaN-safe blurring by extrapolating NaN values
                          before blurring and restoring them afterwards. This prevents
                          NaNs from spreading into valid data during the blur operation.
        """
        from scipy.ndimage import gaussian_filter

        cell_sigma = sigma / self.raster_meta.cell_size

        if preserve_nan:
            # Save the original NaN mask
            nan_mask = np.isnan(self.arr)

            # If there are no NaNs, just apply regular blur
            if not np.any(nan_mask):
                blurred_array = gaussian_filter(self.arr, sigma=cell_sigma)
            else:
                # Extrapolate to fill NaN values temporarily
                extrapolated_arr = fillna_nearest_neighbours(arr=self.arr)

                # Apply blur to the extrapolated array
                blurred_array = gaussian_filter(extrapolated_arr, sigma=cell_sigma)

                # Restore original NaN values
                blurred_array = np.where(nan_mask, np.nan, blurred_array)
        else:
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

    def pad(self, width: float, *, value: float = np.nan) -> Self:
        """Extend the raster by adding a constant fill value around the edges.

        By default, the padding value is NaN, but this can be changed via the
        `value` parameter.

        This grows the raster by adding padding around all edges. New cells are
        filled with the constant `value`.

        If the width is not an exact multiple of the cell size, the padding may be
        slightly larger than the specified width, i.e. the value is rounded up to
        the nearest whole number of cells.

        Args:
            width: The width of the padding, in the same units as the raster CRS
                   (e.g. meters). This defines how far from the edge the padding
                   extends.
            value: The constant value to use for padding. Default is NaN.
        """
        cell_size = self.raster_meta.cell_size

        # Calculate number of cells to pad in each direction
        pad_cells = int(np.ceil(width / cell_size))

        # Get current bounds
        xmin, ymin, xmax, ymax = self.bounds

        # Calculate new bounds with padding
        new_xmin = xmin - (pad_cells * cell_size)
        new_ymin = ymin - (pad_cells * cell_size)
        new_xmax = xmax + (pad_cells * cell_size)
        new_ymax = ymax + (pad_cells * cell_size)

        # Create padded array
        new_height = self.arr.shape[0] + 2 * pad_cells
        new_width = self.arr.shape[1] + 2 * pad_cells

        # Create new array filled with the padding value
        padded_arr = np.full((new_height, new_width), value, dtype=self.arr.dtype)

        # Copy original array into the center of the padded array
        padded_arr[
            pad_cells : pad_cells + self.arr.shape[0],
            pad_cells : pad_cells + self.arr.shape[1],
        ] = self.arr

        # Create new transform for the padded raster
        new_transform = rasterio.transform.from_bounds(
            west=new_xmin,
            south=new_ymin,
            east=new_xmax,
            north=new_ymax,
            width=new_width,
            height=new_height,
        )

        # Create new raster metadata
        new_meta = RasterMeta(
            cell_size=cell_size,
            crs=self.raster_meta.crs,
            transform=new_transform,
        )

        return self.__class__(arr=padded_arr, raster_meta=new_meta)

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
            A new Raster instance cropped to the specified bounds.
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

    def taper_border(self, width: float, *, limit: float = 0.0) -> Self:
        """Taper values to a limiting value around the border of the raster.

        By default, the borders are tapered to zero, but this can be changed via the
        `limit` parameter.

        This keeps the raster size the same, overwriting values in the border area.
        To instead grow the raster, consider using `pad()` followed by `taper_border()`.

        The tapering is linear from the cell centres around the border of the raster,
        so the value at the edge of the raster will be equal to `limit`.

        Args:
            width: The width of the taper, in the same units as the raster CRS
                   (e.g. meters). This defines how far from the edge the tapering
                   starts.
            limit: The limiting value to taper to at the edges. Default is zero.
        """

        # Determine the width in cell units (possibly fractional)
        cell_size = self.raster_meta.cell_size
        width_in_cells = width / cell_size

        # Calculate the distance from the edge in cell units
        arr_height, arr_width = self.arr.shape
        y_indices, x_indices = np.indices((int(arr_height), int(arr_width)))
        dist_from_left = x_indices
        dist_from_right = arr_width - 1 - x_indices
        dist_from_top = y_indices
        dist_from_bottom = arr_height - 1 - y_indices
        dist_from_edge = np.minimum.reduce(
            [dist_from_left, dist_from_right, dist_from_top, dist_from_bottom]
        )

        # Mask the arrays to only the area within the width from the edge, rounding up
        mask = dist_from_edge < np.ceil(width_in_cells)
        masked_dist_arr = np.where(mask, dist_from_edge, np.nan)
        masked_arr = np.where(mask, self.arr, np.nan)

        # Calculate the tapering factor based on the distance from the edge
        taper_factor = np.clip(masked_dist_arr / width_in_cells, 0.0, 1.0)
        tapered_values = limit + (masked_arr - limit) * taper_factor

        # Create the new raster array
        new_arr = self.arr.copy()
        new_arr[mask] = tapered_values[mask]
        new_raster = self.model_copy()
        new_raster.arr = new_arr

        return new_raster

    def clip(
        self,
        polygon: Polygon | MultiPolygon,
        *,
        strategy: Literal["centres"] = "centres",
    ) -> Self:
        """Clip the raster to the specified polygon, replacing cells outside with NaN.

        The clipping strategy determines how to handle cells that are partially
        within the polygon. Currently, only the 'centres' strategy is supported, which
        retains cells whose centres fall within the polygon.

        Args:
            polygon: A shapely Polygon or MultiPolygon defining the area to clip to.
            strategy: The clipping strategy to use. Currently only 'centres' is
                      supported, which retains cells whose centres fall within the
                      polygon.

        Returns:
            A new Raster with cells outside the polygon set to NaN.
        """
        if strategy != "centres":
            msg = f"Unsupported clipping strategy: {strategy}"
            raise NotImplementedError(msg)

        raster = self.model_copy()

        mask = rasterio.features.rasterize(
            [(polygon, 1)],
            fill=0,
            out_shape=self.shape,
            transform=self.meta.transform,
            dtype=np.uint8,
        )

        raster.arr = np.where(mask, raster.arr, np.nan)

        return raster

    def _trim_value(self, *, value_mask: NDArray[np.bool_], value_name: str) -> Self:
        """Crop the raster by trimming away slices matching the mask at the edges.

        Args:
            value_mask: Boolean mask where True indicates values to trim
            value_name: Name of the value type for error messages (e.g., 'NaN', 'zero')
        """
        arr = self.arr

        # Check if the entire array matches the mask
        if np.all(value_mask):
            msg = f"Cannot crop raster: all values are {value_name}"
            raise ValueError(msg)

        # Find rows and columns that are not all matching the mask
        row_mask = np.all(value_mask, axis=1)
        col_mask = np.all(value_mask, axis=0)

        # Find the bounding indices
        (row_indices,) = np.where(~row_mask)
        (col_indices,) = np.where(~col_mask)

        min_row, max_row = row_indices[0], row_indices[-1]
        min_col, max_col = col_indices[0], col_indices[-1]

        # Crop the array
        cropped_arr = arr[min_row : max_row + 1, min_col : max_col + 1]

        # Shift the transform by the number of pixels cropped (min_col, min_row)
        new_transform = (
            self.raster_meta.transform
            * rasterio.transform.Affine.translation(min_col, min_row)
        )

        # Create new metadata
        new_meta = RasterMeta(
            cell_size=self.raster_meta.cell_size,
            crs=self.raster_meta.crs,
            transform=new_transform,
        )

        return self.__class__(arr=cropped_arr, raster_meta=new_meta)

    def trim_nan(self) -> Self:
        """Crop the raster by trimming away all-NaN slices at the edges.

        This effectively trims the raster to the smallest bounding box that contains all
        of the non-NaN values. Note that this does not guarantee no NaN values at all
        around the edges, only that there won't be entire edges which are all-NaN.

        Consider using `.extrapolate()` for further cleanup of NaN values.
        """
        return self._trim_value(value_mask=np.isnan(self.arr), value_name="NaN")

    def trim_zeros(self) -> Self:
        """Crop the raster by trimming away all-zero slices at the edges.

        This effectively trims the raster to the smallest bounding box that contains all
        of the non-zero values. Note that this does not guarantee no zero values at all
        around the edges, only that there won't be entire edges which are all-zero.
        """
        return self._trim_value(value_mask=(self.arr == 0), value_name="zero")

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


def _map_colorbar(
    *,
    colormap: Callable[[float], tuple[float, float, float, float]],
    vmin: float,
    vmax: float,
) -> BrancaLinearColormap:
    from branca.colormap import LinearColormap as BrancaLinearColormap
    from matplotlib.colors import ListedColormap, to_hex

    # Determine legend data range in original units
    vmin = float(vmin) if np.isfinite(vmin) else 0.0
    vmax = float(vmax) if np.isfinite(vmax) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    if isinstance(colormap, ListedColormap):
        n = colormap.N
    else:
        n = 256

    sample_points = np.linspace(0, 1, n)
    colors = [to_hex(colormap(x)) for x in sample_points]
    return BrancaLinearColormap(colors=colors, vmin=vmin, vmax=vmax)


def _get_vmin_vmax(
    raster: Raster, *, vmin: float | None = None, vmax: float | None = None
) -> tuple[float, float]:
    """Get maximum and minimum values from a raster array, ignoring NaNs.

    Allows for custom over-ride vmin and vmax values to be provided.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        if vmin is None:
            _vmin = float(raster.min())
        else:
            _vmin = vmin
        if vmax is None:
            _vmax = float(raster.max())
        else:
            _vmax = vmax

    return _vmin, _vmax


RasterModel = Raster
