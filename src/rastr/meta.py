from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from affine import Affine
from pydantic import BaseModel, InstanceOf
from pyproj import CRS

from rastr.gis.crs import get_affine_sign

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

    @classmethod
    def infer(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        *,
        cell_size: float | None = None,
        crs: CRS,
    ) -> tuple[Self, tuple[int, int]]:
        """Automatically get recommended raster metadata (and shape) using data points.

        The cell size can be provided, or a heuristic will be used based on the spacing
        of the (x, y) points.
        """
        # Heuristic for cell size if not provided
        if cell_size is None:
            cell_size = infer_cell_size(x, y)

        shape = infer_shape(x, y, cell_size=cell_size)
        transform = infer_transform(x, y, cell_size=cell_size, crs=crs)

        raster_meta = cls(
            cell_size=cell_size,
            crs=crs,
            transform=transform,
        )
        return raster_meta, shape


def infer_transform(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cell_size: float | None = None,
    crs: CRS,
) -> Affine:
    """Infer a suitable raster transform based on the bounds of (x, y) data points."""
    if cell_size is None:
        cell_size = infer_cell_size(x, y)

    (xs, ys) = get_affine_sign(crs)
    return Affine.translation(*infer_origin(x, y)) * Affine.scale(
        xs * cell_size, ys * cell_size
    )


def infer_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Infer a suitable raster origin based on the bounds of (x, y) data points."""
    # Compute bounds from data
    minx, _miny, _maxx, maxy = np.min(x), np.min(y), np.max(x), np.max(y)

    origin = (minx, maxy)
    return origin


def infer_shape(
    x: np.ndarray, y: np.ndarray, *, cell_size: float | None = None
) -> tuple[int, int]:
    """Infer a suitable raster shape based on the bounds of (x, y) data points."""
    if cell_size is None:
        cell_size = infer_cell_size(x, y)

    # Compute bounds from data
    minx, miny, maxx, maxy = np.min(x), np.min(y), np.max(x), np.max(y)

    # Compute grid shape
    width = int(np.ceil((maxx - minx) / cell_size))
    height = int(np.ceil((maxy - miny) / cell_size))
    shape = (height, width)

    return shape


def infer_cell_size(x: np.ndarray, y: np.ndarray) -> float:
    """Infer a suitable cell size based on the spacing of (x, y) data points.

    When points are distributed regularly, this corresponds to roughly half the distance
    between neighboring points.

    When distributed irregularly, the size is more influenced by the densest clusters of
    points, i.e. the cell size will be small enough to capture the detail in these
    clusters.

    This is based on a heuristic which has been found to work well in practice.
    """
    from scipy.spatial import KDTree

    # Half the 5th percentile of nearest neighbor distances between the (x,y) points
    xy_points = np.column_stack((x, y))
    tree = KDTree(xy_points)
    distances, _ = tree.query(xy_points, k=2)
    distances: np.ndarray
    cell_size = float(np.percentile(distances[distances > 0], 5)) / 2

    return cell_size
