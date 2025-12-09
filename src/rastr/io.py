"""Deprecated: Use rastr.io_ instead.

This module provides backwards compatibility by re-exporting all functions from io_.
"""  # noqa: A005

import warnings

from rastr.io_ import (
    read_cad_gdf,
    read_raster_inmem,
    read_raster_mosaic_inmem,
    write_raster,
)

warnings.warn(
    "rastr.io is deprecated. Import from rastr.io_ instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "read_cad_gdf",
    "read_raster_inmem",
    "read_raster_mosaic_inmem",
    "write_raster",
]
