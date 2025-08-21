from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from affine import Affine
from pydantic import BaseModel, InstanceOf
from pyproj import CRS

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing_extensions import Self


class RasterMeta(BaseModel, extra="forbid"):
    """Raster metadata.

    Attributes:
        cell_size: Cell size in meters.
        crs: Coordinate reference system.
        transform: The affine transformation associated with the raster. This is based
                   on the CRS, the cell size, as well as the offset/origin.
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

    def get_cell_centre_coords(self, shape: tuple[int, int]) -> NDArray:
        """Return an array of (x, y) coordinates for the center of each cell.

        The coordinates will be in the coordinate system defined by the
        raster's transform.

        Args:
            shape: (rows, cols) of the raster array.

        Returns:
            (x, y) coordinates for each cell center, with shape (rows, cols, 2)
        """
        x_coords = self.get_cell_x_coords(shape[1])  # cols for x-coordinates
        y_coords = self.get_cell_y_coords(shape[0])  # rows for y-coordinates
        coords = np.stack(np.meshgrid(x_coords, y_coords), axis=-1)
        return coords

    def get_cell_x_coords(self, n_columns: int) -> NDArray:
        """Return an array of x coordinates for the center of each cell.

        The coordinates will be in the coordinate system defined by the
        raster's transform.

        Args:
            n_columns: Number of columns in the raster array.

        Returns:
            x_coordinates at cell centers, with shape (n_columns,)
        """
        x_idx = np.arange(n_columns) + 0.5
        y_idx = np.zeros_like(x_idx)  # Use y=0 for a single row
        x_coords, _ = self.transform * (x_idx, y_idx)  # type: ignore[reportAssignmentType] overloaded tuple size in affine
        return x_coords

    def get_cell_y_coords(self, n_rows: int) -> NDArray:
        """Return an array of y coordinates for the center of each cell.

        The coordinates will be in the coordinate system defined by the
        raster's transform.

        Args:
            n_rows: Number of rows in the raster array.

        Returns:
            y_coordinates at cell centers, with shape (n_rows,)
        """
        x_idx = np.zeros(n_rows)  # Use x=0 for a single column
        y_idx = np.arange(n_rows) + 0.5
        _, y_coords = self.transform * (x_idx, y_idx)  # type: ignore[reportAssignmentType] overloaded tuple size in affine
        return y_coords
