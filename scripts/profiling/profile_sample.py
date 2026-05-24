"""Profile the Raster.sample() method using pyinstrument.

This script profiles the `.sample()` method on a Raster object with varying sizes
of input data to understand performance characteristics.

Usage:
    uv run python scripts/profiling/profile_sample.py

Output:
    - HTML profiling reports in scripts/profiling/
    - Reports can be opened in a web browser for interactive visualization

For more information on optimizing sampling, see:
https://rdrn.me/optimising-sampling/
"""

from pathlib import Path

import numpy as np
from affine import Affine
from pyinstrument import Profiler
from pyproj.crs.crs import CRS

from rastr.meta import RasterMeta
from rastr.raster import Raster


def create_test_raster(size: int = 1000) -> Raster:
    """Create a test raster of given size.

    Args:
        size: The size of the square raster (size x size).

    Returns:
        A Raster object with random data.
    """
    rng = np.random.default_rng()
    meta = RasterMeta(
        cell_size=1.0,
        crs=CRS.from_epsg(2193),
        transform=Affine(1.0, 0.0, 0.0, 0.0, -1.0, float(size)),
    )
    arr = rng.random((size, size))
    return Raster(arr=arr, raster_meta=meta)


def profile_sample_points(raster: Raster, num_points: int) -> None:
    """Profile sampling a number of points from a raster.

    Args:
        raster: The raster to sample from.
        num_points: Number of points to sample.
    """
    rng = np.random.default_rng()
    # Generate random points within the raster bounds
    bounds = raster.bounds
    x_coords = rng.uniform(bounds.xmin, bounds.xmax, num_points)
    y_coords = rng.uniform(bounds.ymin, bounds.ymax, num_points)
    points = list(zip(x_coords, y_coords, strict=True))

    # Profile the sampling operation
    profiler = Profiler()
    profiler.start()

    _ = raster.sample(points)

    profiler.stop()

    # Save the profiling report
    output_file = Path(f"scripts/profiling/sample_{num_points}_points.html")
    output_file.write_text(profiler.output_html())
    print(f"Profiling report saved to: {output_file}")


if __name__ == "__main__":
    print("Creating test raster (1000x1000)...")
    test_raster = create_test_raster(size=1000)

    print("\nProfiling sample() with different numbers of points...\n")

    # Profile with increasing numbers of sample points
    for num_points in [100, 1000, 10000]:
        print(f"Profiling {num_points} sample points...")
        profile_sample_points(test_raster, num_points)

    print("\nProfiling complete! Open the HTML files in a web browser to view results.")
