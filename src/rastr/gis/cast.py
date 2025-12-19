from __future__ import annotations

from typing import TYPE_CHECKING

from shapely.geometry import LineString, MultiLineString

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


def cast_multilinestring(geom: BaseGeometry) -> MultiLineString:
    """Cast a geometry to a MultiLineString.

    If the geometry is a LineString, it will be converted to a single-part
    MultiLineString. MultiLineStrings are returned unchanged.

    Args:
        geom: The geometry to cast.
    """
    if geom.is_empty:
        return MultiLineString()

    if geom.geom_type == "MultiLineString":
        if not isinstance(geom, MultiLineString):
            raise AssertionError
        return geom
    elif geom.geom_type == "LineString":
        if not isinstance(geom, LineString):
            raise AssertionError
        return MultiLineString([geom])
    else:
        msg = f"Cannot cast geometry of type {geom.geom_type} to MultiLineString."
        raise TypeError(msg)
