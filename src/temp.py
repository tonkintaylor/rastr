from pyproj.crs.crs import CRS
from rasterio.transform import from_origin

from rastr import Raster, RasterMeta
from rastr.create import full_raster

# Create an example raster
raster = Raster.example()

# Write to and read from a file
raster.to_file("raster.tif")
raster = Raster.from_file("raster.tif")

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
