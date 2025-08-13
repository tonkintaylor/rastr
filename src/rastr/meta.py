import numpy as np
from affine import Affine
from pydantic import BaseModel, InstanceOf
from pyproj import CRS
from typing_extensions import Self


class RasterMeta(BaseModel, extra="forbid"):
    """Raster metadata.

    Attributes:
        cell_size: Cell size in meters.
        crs: Coordinate reference system.
    """

    cell_size: float
    crs: InstanceOf[CRS]
    transform: InstanceOf[Affine]

    @classmethod
    def example(cls) -> Self:
        """Create an example RasterMeta object."""
        return cls(
            cell_size=2.0,
            crs=CRS.from_epsg(2193),
            transform=Affine.scale(2.0, 2.0),
        )

    def get_cell_centre_coords(self, shape: tuple[int, int]) -> np.ndarray:
        """Return an array of (x, y) coordinates for the center of each cell.

        The coordinates will be in the coordinate system defined by the
        raster's transform.

        Args:
            shape: (rows, cols) of the raster array.

        Returns:
            np.ndarray of shape (rows, cols, 2) with (x, y) coordinates for each
            cell center.
        """
        rows, cols = shape
        x_idx = np.arange(cols)
        y_idx = np.arange(rows)
        xv, yv = np.meshgrid(x_idx, y_idx)
        x_coords, y_coords = self.transform * (xv + 0.5, yv + 0.5)
        coords = np.stack([x_coords, y_coords], axis=-1)
        return coords
