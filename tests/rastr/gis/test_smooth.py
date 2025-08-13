import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from rastr.gis.smooth import _recursive_eval, catmull_rom_smooth


class TestCatmullRomSmooth:
    def test_linestring(self):
        # Arrange
        ls = LineString([(0, 0), (1, 1), (2, 0)])
        # Act
        result = catmull_rom_smooth(ls, alpha=0.5, subdivs=5)
        # Assert
        assert isinstance(result, LineString)
        assert len(result.coords) > len(ls.coords)
        assert result.coords[0] == pytest.approx(ls.coords[0])
        assert result.coords[-1] == pytest.approx(ls.coords[-1])

    def test_linestring_closed(self):
        # Arrange
        coords = [(0, 0), (1, 2), (2, 0), (0, 0)]
        ls = LineString(coords)
        # Act
        result = catmull_rom_smooth(ls, alpha=0.5, subdivs=8)
        # Assert
        assert isinstance(result, LineString)
        assert result.coords[0] == pytest.approx(result.coords[-1])

    def test_polygon(self):
        # Arrange
        poly = Polygon([(0, 0), (2, 0), (1, 2), (0, 0)])
        # Act
        result = catmull_rom_smooth(poly, alpha=0.5, subdivs=6)
        # Assert
        assert isinstance(result, Polygon)
        assert len(result.interiors) == 0
        assert result.exterior.coords[0] == pytest.approx(result.exterior.coords[-1])

    def test_polygon_with_hole(self):
        # Arrange
        exterior = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
        hole = [(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]
        poly = Polygon(exterior, [hole])
        # Act
        result = catmull_rom_smooth(poly, alpha=0.5, subdivs=4)
        # Assert
        assert isinstance(result, Polygon)
        assert len(result.interiors) == 1
        assert result.exterior.coords[0] == pytest.approx(result.exterior.coords[-1])
        assert result.interiors[0].coords[0] == pytest.approx(
            result.interiors[0].coords[-1]
        )

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("subdivs", [2, 5, 10])
    def test_parameter_variations(self, alpha, subdivs):
        # Arrange
        ls = LineString([(0, 0), (1, 2), (2, 0)])
        # Act
        result = catmull_rom_smooth(ls, alpha=alpha, subdivs=subdivs)
        # Assert
        assert isinstance(result, LineString)
        assert len(result.coords) > len(ls.coords)

    @pytest.mark.parametrize(
        ("alpha", "subdivs", "expected"),
        [
            (
                0.0,
                2,
                [(0.0, 0.0), (0.375, 0.5), (1.0, 1.0), (1.4375, 0.5625), (2.0, 0.0)],
            ),
            (
                0.5,
                2,
                [
                    (0.0, 0.0),
                    (0.375, 0.5),
                    (1.0, 1.0),
                    (1.46107079, 0.5679017),
                    (2.0, 0.0),
                ],
            ),
            (
                1.0,
                2,
                [
                    (0.0, 0.0),
                    (0.375, 0.5),
                    (1.0, 1.0),
                    (1.47855339, 0.5732233),
                    (2.0, 0.0),
                ],
            ),
            (
                0.0,
                5,
                [
                    (0.0, 0.0),
                    (0.072, 0.104),
                    (0.256, 0.352),
                    (0.504, 0.648),
                    (0.768, 0.896),
                    (1.0, 1.0),
                    (1.184, 0.912),
                    (1.352, 0.696),
                    (1.528, 0.424),
                    (1.736, 0.168),
                    (2.0, 0.0),
                ],
            ),
            (
                0.5,
                5,
                [
                    (0.0, 0.0),
                    (0.072, 0.104),
                    (0.256, 0.352),
                    (0.504, 0.648),
                    (0.768, 0.896),
                    (1.0, 1.0),
                    (1.19003412, 0.91338284),
                    (1.37010237, 0.70014851),
                    (1.55515355, 0.43022276),
                    (1.76013649, 0.17353134),
                    (2.0, 0.0),
                ],
            ),
            (
                1.0,
                5,
                [
                    (0.0, 0.0),
                    (0.072, 0.104),
                    (0.256, 0.352),
                    (0.504, 0.648),
                    (0.768, 0.896),
                    (1.0, 1.0),
                    (1.19450967, 0.91474517),
                    (1.383529, 0.7042355),
                    (1.57529351, 0.43635325),
                    (1.77803867, 0.17898066),
                    (2.0, 0.0),
                ],
            ),
        ],
    )
    def test_regression(self, alpha, subdivs, expected):
        # Arrange
        ls = LineString([(0, 0), (1, 1), (2, 0)])
        # Act
        result = catmull_rom_smooth(ls, alpha=alpha, subdivs=subdivs)
        # Assert
        coords = [(round(x, 8), round(y, 8)) for x, y in result.coords]
        assert coords == expected


class TestRecursiveEval:
    def test_denom_zero(self):
        # Arrange
        # Create a degenerate case where tangents are all the same (i.e. co-located
        # points)
        coords = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        # This will force denom == 0 in _recursive_eval for the first segment

        # Act
        result = _recursive_eval(coords, [0.0, 0.0, 1.0, 2.0], np.array([0.0]))

        # Assert
        assert isinstance(result, list)
        assert all(isinstance(pt, tuple) and len(pt) == 2 for pt in result)
