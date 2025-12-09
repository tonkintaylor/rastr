<h1 align="center">
  <img src="https://raw.githubusercontent.com/tonkintaylor/rastr/refs/heads/develop/docs/logo.svg"><br>
</h1>

# rastr

[![PyPI Version](https://img.shields.io/pypi/v/rastr.svg)](<https://pypi.python.org/pypi/rastr>)
[![PyPI Supported Versions](https://img.shields.io/pypi/pyversions/rastr.svg)](https://pypi.python.org/pypi/rastr)
![PyPI License](https://img.shields.io/pypi/l/rastr.svg)

A lightweight geospatial raster datatype library for Python focused on simplicity.

For more details, read the documentation: <https://rastr.readthedocs.io/en/stable/>.

## Overview

`rastr` provides an intuitive interface for creating, reading, manipulating, and exporting geospatial raster data in Python.

### Features

- 🧮 **Complete raster arithmetic**: Full support for mathematical operations (`+`, `-`, `*`, `/`) between rasters and scalars.
- 📊 **Flexible visualization**: Built-in plotting with matplotlib and interactive mapping with folium.
- 🗺️ **Geospatial analysis tools**: Contour generation, Gaussian blurring, and spatial sampling.
- 🛠️ **Data manipulation**: Fill NaN values, extrapolate missing data, and resample to different resolutions.
- 🔗 **Seamless integration**: Works with GeoPandas, rasterio, and the broader Python geospatial ecosystem.
- ↔️ **Vector-to-raster workflows**: Convert GeoDataFrame polygons, points, and lines to raster format.

## Installation

<!--pytest.mark.skip-->
```bash
# With uv
uv add rastr

# With pip
pip install rastr
```

## Quick Start

```python
from pyproj.crs.crs import CRS
from rasterio.transform import from_origin
from rastr import Raster, RasterMeta
from rastr.create import full_raster


# Create an example raster
raster = Raster.example()

# Write to and read from a file
raster.to_file("raster.tif")
raster = Raster.read_file("raster.tif")

# Basic arithmetic operations
doubled = raster * 2
summed = raster + 10
combined = raster + doubled

# Create full rasters with specified values
cell_size = 1.0
empty_raster = full_raster(
    RasterMeta(
        cell_size=cell_size,
        crs=CRS.from_epsg(2193),
        transform=from_origin(0, 100, cell_size, cell_size),
    ),
    bounds=(0, 0, 100, 100),
    fill_value=0.0,
)

# Visualize the data
ax = raster.plot(cbar_label="Values")

# Interactive web mapping (requires folium)
m = raster.explore(opacity=0.8, colormap="plasma")

# Sample values at specific coordinates
xy_points = [(100.0, 200.0), (150.0, 250.0)]
values = raster.sample(xy_points)

# Generate contour lines
contours = raster.contour(levels=[0.1, 0.5, 0.9], smoothing=True)

# Apply spatial operations
blurred = raster.blur(sigma=2.0)  # Gaussian blur
filled = raster.extrapolate(method="nearest")  # Fill NaN values via nearest-neighbours
resampled = raster.resample(cell_size=0.5)  # Change resolution

# Export to file
raster.to_file("output.tif")

# Convert to GeoDataFrame for vector analysis
gdf = raster.as_geodataframe(name="elevation")
```

## Quick Reference

```python
from rastr import Raster
```

### Data access

- [`Raster.bbox`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.bbox) - bounding box polygon.
- [`Raster.bounds`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.bounds) - bounding box as `(xmin, ymin, xmax, ymax)`.
- [`Raster.cell_size`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.cell_size) - cell size.
- [`Raster.crs`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.crs) - coordinate reference system.
- [`Raster.shape`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.shape) - raster shape (rows, columns).
- [`Raster.transform`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.transform) - affine transform.
- [`Raster.sample(xy)`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.sample) - sample raster values at given coordinates.

### I/O

- [`Raster.read_file(path)`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.read_file) - read raster from file.
- [`Raster.to_file(path)`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.to_file) - write raster to file.
- [`Raster.to_clipboard()`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.to_clipboard) - copy raster data to clipboard in a tabular format.

### Geometric Operations

- [`Raster.crop(bounds)`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.crop) - remove cells outside given bounds.
- [`Raster.pad(width)`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.pad) - add NaN border around raster.
- [`Raster.resample(cell_size)`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.resample) - resample raster to a new cell size.
- [`Raster.taper_border(width)`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.taper_border) - gradually reduce values to zero at the border.
- [`Raster.gdf()`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.gdf) - Vectorize to a GeoDataFrame of cell polygons and values.

### NaN Management and value replacements

- [`Raster.clip(polygon)`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.clip) - replace values outside a polygon with NaN.
- [`Raster.extrapolate()`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.extrapolate) - fill NaN values via nearest-neighbours.
- [`Raster.fillna(value)`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.fillna) - fill NaN values with a specified value.
- [`Raster.replace(to_replace, value)`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.replace) - replace specific cell values.
- [`Raster.replace_polygon(polygon, value)`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.replace_polygon) - replace cell values within a polygon.
- [`Raster.trim_nan()`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.trim_nan) - remove rows/columns that are entirely NaN.

### Image Processing

- [`Raster.blur(radius)`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.blur) - apply Gaussian blur.
- [`Raster.dilate(radius)`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.dilate) - apply morphological dilation.
- [`Raster.sobel()`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.sobel) - apply Sobel filter (edge detection/gradient).

### Visualization

- [`Raster.explore()`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.explore) - interactive web map visualization with folium.
- [`Raster.plot()`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.plot) - matplotlib static plot with colorbar.
- [`Raster.contour(levels)`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.contour) - get a GeoDataFrame of contour lines.

### Cell-wise Operations

- [`Raster.apply(func)`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.apply) - apply a function to cell values.
- [`Raster.abs()`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.abs) - absolute value of cell values.
- [`Raster.exp()`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.exp) - exponential of cell values.
- [`Raster.log()`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.log) - logarithm of cell values.
- [`Raster.max()`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.max) - maximum of cell values.
- [`Raster.mean()`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.mean) - mean of cell values.
- [`Raster.median()`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.median) - median of cell values.
- [`Raster.min()`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.min) - minimum of cell values.
- [`Raster.normalize()`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.normalize) - normalize cell values to [0, 1].
- [`Raster.quantile(q)`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.quantile) - quantile of cell values.
- [`Raster.std()`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.std) - standard deviation of cell values.
- [`Raster.sum()`](https://rastr.readthedocs.io/en/stable/autoapi/rastr/raster/#rastr.raster.Raster.sum) - sum of cell values.

## Limitations

Current version limitations:

- Only Single-band rasters are supported.
- In-memory processing only (streaming support planned).
- Square cells only (rectangular cell support planned).
- Only float dtypes (integer support planned).

## Similar Projects

- [rasters](https://github.com/python-rasters/rasters) is a project with similar goals of providing a dedicated raster datatype in Python with higher-level interfaces for GIS operations. Unlike `rastr`, it has support for multi-band rasters, and has some more advanced functionality for Earth Science applications. Both projects are relatively new and under active development.
- [rasterio](https://rasterio.readthedocs.io/) is a core dependency of `rastr` and provides low-level raster I/O and processing capabilities.
- [rioxarray](https://corteva.github.io/rioxarray/stable/getting_started/getting_started.html) extends [`xarray`](https://docs.xarray.dev/en/stable/index.html) for raster data with geospatial support via `rasterio`.

### Contributing

See the
[CONTRIBUTING.md](https://github.com/usethis-python/usethis-python/blob/main/CONTRIBUTING.md)
file.
