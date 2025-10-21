from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self

from pyproj import CRS

import rastr.gis.crs
from rastr.gis.crs import get_affine_sign

if TYPE_CHECKING:
    import pytest


class TestGetAffineSign:
    def test_wgs84(self):
        crs = "EPSG:4326"  # lat/lon, north up
        x_sign, y_sign = get_affine_sign(crs)
        assert x_sign == 1
        assert y_sign == -1

    def test_nztm(self):
        crs = "EPSG:2193"
        x_sign, y_sign = get_affine_sign(crs)
        assert x_sign == 1
        assert y_sign == -1

    def test_web_mercator(self):
        crs = "EPSG:3857"  # Web Mercator, axis directions not specified
        x_sign, y_sign = get_affine_sign(crs)
        assert x_sign == 1
        assert y_sign == -1

    def test_mock(self, monkeypatch: pytest.MonkeyPatch):
        # Can't find any EPSG codes that use non-standard axis directions,
        # so we have to monkeypatch the axis_info direction manually.
        crs = CRS.from_user_input("EPSG:4326")

        class MockCRS:
            @classmethod
            def from_user_input(cls, _: Any) -> Self:
                return cls()

            axis_info: ClassVar = [
                type("Axis", (), {"direction": "west"}),
                type("Axis", (), {"direction": "south"}),
            ]

        monkeypatch.setattr(rastr.gis.crs, "CRS", MockCRS)
        x_sign, y_sign = get_affine_sign(crs)

        assert x_sign == -1
        assert y_sign == 1
