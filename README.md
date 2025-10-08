<h1 align="center">
  <img src="https://raw.githubusercontent.com/tonkintaylor/rastr/refs/heads/develop/docs/logo.svg"><br>
</h1>

# rastr

[![PyPI Version](https://img.shields.io/pypi/v/rastr.svg)](<https://pypi.python.org/pypi/rastr>)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![usethis](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/usethis-python/usethis-python/main/assets/badge/v1.json)](https://github.com/usethis-python/usethis-python)

A lightweight geospatial raster datatype library for Python focused on simplicity.

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
from rastr.create import full_raster
from rastr.meta import RasterMeta
from rastr.raster import Raster

# Read a raster from a file
raster = Raster.read_file("path/to/raster.tif")

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
resampled = raster.resample(new_cell_size=0.5)  # Change resolution

# Export to file
raster.to_file("output.tif")

# Convert to GeoDataFrame for vector analysis
gdf = raster.as_geodataframe(name="elevation")
```

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
